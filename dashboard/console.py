"""Streamlit operator console for local aggregate observations."""
from bisect import bisect_right
from html import escape
import json
import os
from pathlib import Path
import time

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.camera_map import camera_plan
from dashboard.data import DEFAULT_STATE, live_state, replays

st.set_page_config(page_title="TransitSense · Operations", page_icon=":material/directions_bus:", layout="wide")
st.html(Path(__file__).with_name("console.css"))
ss = st.session_state

@st.cache_data(max_entries=20, show_spinner=False)
def load_report(path, modified):
    report = json.loads(path.read_bytes())
    times = [sample["t"] for sample in report["samples"]]
    if not times or any(a >= b for a,b in zip(times,times[1:])):
        raise ValueError("Replay needs observations in time order.")
    return report, times


def toggle_play(end):
    now = time.monotonic()
    if ss.playing:
        ss.cursor = min(end, ss.origin_position + now - ss.origin_time)
    else:
        if ss.cursor >= end:
            ss.cursor = 0.0
        ss.origin_position, ss.origin_time = ss.cursor, now
    ss.playing = not ss.playing


def seek():
    ss.cursor = ss.scrubber
    ss.playing = False


def source_changed():
    ss.playing = False
    ss.cursor = 9.5 if ss.source == "controlled" else 0.0
    ss.pop("scrubber", None)


def readings(sample):
    flow = sample.get("flow") or {}
    states = sample.get("seats") or {}
    metrics = [
        ("People in view", sample.get("visible"), "groups"),
        ("Boardings", flow.get("boarding"), "login"),
        ("Alightings", flow.get("alighting"), "logout"),
        ("Seats marked available", sum(v == "available" for v in states.values()) if states else None, "event_seat"),
    ]
    with st.container(key="readings"):
        for col,(name,value,icon) in zip(st.columns(4), metrics):
            col.metric(name, "—" if value is None else value, icon=f":material/{icon}:", border=True)


def zones_panel(sample):
    st.subheader("Passenger distribution")
    zones = sample.get("zones", {})
    for name in ("front", "middle", "rear"):
        zone = zones.get(name, {})
        count = zone.get("people")
        status = zone.get("status")
        description = {"observed":"Visible people","uncertain":"Limited confidence"}.get(status, "Outside measured coverage")
        if sample.get("quality") not in (None, "usable"):
            description = "Visibility too low"
        remaining = zone.get("remaining")
        if remaining is not None:
            description += f" · {remaining} places remaining"
        st.html(f'<div class="zone-row"><div><strong>{name.title()}</strong><small>{escape(description)}</small></div>'
                f'<b>{escape(str(count)) if count is not None else "—"}</b></div>')


def passage_panel(sample):
    passages = sample.get("passages", {})
    st.subheader("Passage monitoring")
    if not passages:
        st.caption("No monitored passage in this view.")
    for name,p in passages.items():
        status = p["status"]
        message = {"review":"Possible obstruction","monitoring":"Object present","no candidate":"No candidate detected",
                   "unobserved":"Outside measured coverage"}.get(status, "Unavailable")
        dwell = p.get("dwell_s", 0)
        if dwell:
            message += f" · {dwell:.1f} s"
        css = "alert-line" if status=="review" else "passage-line"
        st.html(f'<div class="{css}"><strong>{escape(name.title())}</strong> · {escape(message)}</div>')


def details(data, sample):
    with st.expander("Seat register & run details", icon=":material/tune:"):
        left,right = st.columns(2)
        seats=sample.get("seats",{})
        if seats:
            left.dataframe([{"Seat":k,"State":v.title()} for k,v in seats.items()], hide_index=True,
                           width="stretch")
        else:
            left.caption("No mapped seat observations.")
        stats = data.get("benchmark") or sample.get("performance",{})
        items = [(label, f"{stats[key]}{unit}") for key,label,unit in
                 [("frames","Frames",""),("processing_fps","Throughput"," FPS"),("p95_ms","Processing p95"," ms"),
                  ("peak_rss_mib","Peak memory"," MiB")] if key in stats]
        if (data.get("door") or {}).get("max_gap_s") is not None:
            items.append(("Track memory",f'{data["door"]["max_gap_s"]} s'))
        if data.get("layout",{}).get("dwell_s"):
            items.append(("Review threshold",f'{data["layout"]["dwell_s"]} s'))
        if items:
            right.dataframe([{"Measurement":k,"Value":v} for k,v in items],hide_index=True,width="stretch")


def observation(data, sample):
    readings(sample)
    main,side = st.columns([1.8, 1], gap="medium")
    with main, st.container(border=True):
        st.subheader("Camera plan")
        layout = data.get("layout")
        if layout:
            st.image(camera_plan(layout,data.get("door"),sample), width="stretch")
            st.html('<div class="legend"><span>Occupied</span><span>Available</span><span>Uncertain</span><span>Door line</span></div>')
        else:
            st.caption("Waiting for camera calibration.")
    with side, st.container(border=True):
        zones_panel(sample)
        passage_panel(sample)


extra = os.environ.get("TRANSITSENSE_REPLAY")
sources = replays(Path(extra) if extra else None)
live_file = Path(os.environ.get("TRANSITSENSE_STATE") or DEFAULT_STATE)
if "cursor" not in ss:
    ss.cursor, ss.playing = 9.5, False

st.html('<div class="mast"><span class="route-mark" aria-hidden="true">≋</span><b>TransitSense</b>'
        '<span class="mast-divider"></span><span>Operator console</span></div>')
title, source_col = st.columns([1.5,1],vertical_alignment="bottom")
title.title("Service overview")
source_col.selectbox("Measurement source", [*sources,"live"], key="source",
                     format_func=lambda k: "Live monitor" if k=="live" else sources[k][0], on_change=source_changed)

if ss.source == "live":
    @st.fragment(run_every=1.0)
    def live_view():
        try:
            data = live_state(live_file)
        except (ValueError, OSError, KeyError):
            st.error("Live measurements could not be read.")
            return
        state = data.get("status")
        label = {"running":"Receiving updates","ended":"Run ended · final observation","starting":"Starting camera",
                 "stale":"Connection stale","offline":"No monitor connected","interrupted":"Run interrupted"}.get(state,"Unavailable")
        sample = data.get("sample") if state in ("running", "ended") else None
        if sample:
            st.caption(f'{label}  ·  {data.get("name","Camera")}  ·  {sample["t"]:.2f} s  ·  Image detail: {sample.get("quality","unknown")}')
            observation(data, sample)
            details(data, sample)
        else:
            st.info(label, icon=":material/sensors:")
    live_view()
else:
    path=sources[ss.source][1]
    try:
        data,times=load_report(path,path.stat().st_mtime_ns)
    except (ValueError, KeyError, OSError) as error:
        st.error(f"Replay unavailable: {error}")
        st.stop()
    end=float(times[-1])

    @st.fragment(run_every=.5 if ss.playing else None)
    def replay_view():
        if ss.playing:
            ss.cursor = min(end,ss.origin_position + time.monotonic() - ss.origin_time)
            if ss.cursor >= end:
                ss.playing=False
                st.rerun()
        position=min(float(ss.cursor),end)
        sample=data["samples"][max(0,bisect_right(times,position)-1)]
        st.caption(f'{data["source"]}  ·  Replay  ·  {sample["t"]:.2f} s  ·  Image detail: {sample["quality"]}')
        observation(data,sample)
        with st.container(border=True):
            st.subheader("Replay timeline")
            play, slider=st.columns([1,5],vertical_alignment="bottom")
            if play.button("Pause" if ss.playing else "Play", icon=":material/pause:" if ss.playing else ":material/play_arrow:",
                           key="play",type="primary",width="stretch",disabled=end<=0):
                toggle_play(end)
                st.rerun()
            if end>0:
                ss.scrubber=position
                slider.slider("Source time",0.0,end,step=.01,format="%.2f s",key="scrubber",on_change=seek)
            if len(times)>1:
                trend = pd.DataFrame({"Source time (s)":times,"People in view":[s.get("visible") for s in data["samples"]]})
                people = alt.Chart(trend).mark_area(
                    interpolate="step-after", color="#9ac4a7", line={"color": "#246a55"}
                ).encode(x=alt.X("Source time (s):Q"), y=alt.Y("People in view:Q", axis=alt.Axis(tickMinStep=1)))
                clock = alt.Chart(pd.DataFrame({"position": [position]})).mark_rule(
                    color="#985619", strokeDash=[4, 3]
                ).encode(x="position:Q")
                st.altair_chart((people + clock).properties(height=145), width="stretch")
        details(data, sample)
    replay_view()
