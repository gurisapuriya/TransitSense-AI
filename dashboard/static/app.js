"use strict";
const $ = id => document.getElementById(id);
const ns = "http://www.w3.org/2000/svg";
const colors = {available:"#68d7a4", occupied:"#61b9ce", uncertain:"#97a9ad"};
let run = null, selection = "", playing = false, position = 0, previousTick = 0;
let generation = 0, polling = false, lastSample = -1;

function tag(name, text, css) {
  const el = document.createElement(name);
  if (text !== undefined) el.textContent = text;
  if (css) el.className = css;
  return el;
}
function shape(name, attrs, text) {
  const el = document.createElementNS(ns,name);
  for (const [key,val] of Object.entries(attrs)) el.setAttribute(key,val);
  if (text) el.textContent = text;
  $("map").append(el);
}
function geometry(layout, sample, door) {
  $("map").replaceChildren();
  const width = 760, height = width / (layout?.aspect_ratio || 2);
  $("map").setAttribute("viewBox",[-16,-16,width+32,height+32].join(" "));
  function polygon(p, fill, stroke, dash) {
    shape("polygon",{points:p.map(([x,y])=>[x*width,y*height].join(",")).join(" "),fill,stroke,
      "stroke-width":2,"stroke-dasharray":dash || "none"});
  }
  for (const [k,p] of Object.entries(layout?.zones || {})) {
    if (!p) continue;
    polygon(p, {front:"#275963",middle:"#1c4957",rear:"#214252"}[k] || "#214252", "#74989f");
    const [x,y] = p[0];
    shape("text",{x:x*width+5,y:y*height+17,fill:"#e6fffa","font-size":15},k.toUpperCase());
  }
  for (const [k,p] of Object.entries(layout?.passages || {})) {
    if (!p) continue;
    const alert = sample?.passages[k]?.status === "review";
    polygon(p,alert?"#935b30":"none",alert?"#ffc482":"#71a3a5","5 5");
  }
  for (const [k,r] of Object.entries(layout?.seats || {})) {
    shape("rect",{x:r[0]*width,y:r[1]*height,width:(r[2]-r[0])*width,height:(r[3]-r[1])*height,
      rx:4,fill:colors[sample?.seats[k]] || colors.uncertain,"fill-opacity":.85,stroke:"#163e48"});
    shape("text",{x:(r[0]+r[2])*width/2,y:(r[1]+r[3])*height/2+5,"text-anchor":"middle",
      fill:"#102d39","font-size":13,"font-weight":700},k);
  }
  if (door) {
    shape("line",{x1:door.start[0]*width,y1:door.start[1]*height,x2:door.end[0]*width,
      y2:door.end[1]*height,stroke:"#ffb36d","stroke-width":5});
    if (Object.keys(layout?.seats || {}).length === 0) {
      shape("text",{x:door.inside[0]*width,y:door.inside[1]*height,fill:"#e5f2f2","text-anchor":"middle",
        "font-size":22},"INTERIOR SIDE");
    }
  }
  if (!layout) shape("text",{x:width/2,y:height/2,fill:"#d6e7e7","text-anchor":"middle","font-size":21},"Waiting for measurements");
}
function row(parent, name, description, value, alert=false) {
  const r=tag("div",undefined,"row"), left=tag("div",name);
  if (description) left.append(tag("small",description));
  r.append(left,tag("span",value,alert?"value attention":"value")); parent.append(r);
}
function render(data,s,state) {
  const flow=s?.flow, mapped=Object.entries(s?.seats || {});
  $("visible").textContent=s?.visible ?? "—";
  $("boarding").textContent=flow?.boarding ?? "—"; $("alighting").textContent=flow?.alighting ?? "—";
  $("available").textContent=mapped.length ? mapped.filter(([,v])=>v==="available").length : "—";
  $("source").textContent=data ? [data.name,data.source,state].filter(Boolean).join("  ·  ") : state;
  $("quality").textContent=s ? "Image detail · "+s.quality : "Unavailable";
  geometry(data?.layout,s,data?.door);
  $("zones").replaceChildren();
  for (const k of ["front","middle","rear"]) {
    const z=s?.zones[k];
    const info=z?.status==="uncertain"?"Limited confidence":z?.status==="observed"?"Visible people":s && s.quality!=="usable"?"Visibility too low":"Outside measured coverage";
    row($("zones"),k[0].toUpperCase()+k.slice(1),z?.remaining!=null?info+" · "+z.remaining+" places remaining":info,z?.people??"—");
  }
  $("passages").replaceChildren();
  const passages=Object.entries(s?.passages || {});
  for (const [k,p] of passages) {
    const hint={"review":"Possible obstruction","monitoring":"Object present","no candidate":"No candidate detected","unobserved":"Outside measured coverage"}[p.status];
    row($("passages"),k[0].toUpperCase()+k.slice(1),hint,p.status==="review"?"Review · "+p.dwell_s.toFixed(1)+" s":p.dwell_s?p.dwell_s.toFixed(1)+" s":"—",p.status==="review");
  }
  if (!passages.length) $("passages").append(tag("p","No passage measurements for this view.","muted"));
  $("seats").replaceChildren(); $("seatTotal").textContent=mapped.length?mapped.length+" positions":"";
  for (const [k,v] of mapped) $("seats").append(tag("span",k+" · "+v,"seat "+v));
  if (!mapped.length) $("seats").append(tag("span","No seat measurements for this view.","muted"));
  $("details").replaceChildren();
  function detail(k,v){const div=tag("div");div.append(tag("dt",k),tag("dd",v));$("details").append(div);}
  detail("Source time",s?s.t.toFixed(2)+" s":"—");
  const b=data?.benchmark || s?.performance;
  if (b) {
    detail("Processed frames",String(b.frames));detail("Throughput",b.processing_fps+" FPS");
    detail("Processing p95",b.p95_ms+" ms");detail("Peak process memory",b.peak_rss_mib+" MiB");
  }
  if (data?.door) detail("Track memory",data.door.max_gap_s+" s");
  if (data?.layout?.dwell_s) detail("Obstruction review after",data.layout.dwell_s+" s");
}
function setPlay(on){playing=on;previousTick=performance.now();$("play").textContent=on?"Pause":"Play";}
function drawReplay() {
  if (!run) return;
  const samples=run.samples;
  let lo=0,hi=samples.length-1;
  while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(samples[mid].t<=position)lo=mid;else hi=mid-1;}
  $("seek").value=position; $("time").textContent=position.toFixed(1)+" s";
  if(lo!==lastSample){render(run,samples[lo],"Replay");lastSample=lo;}
}
async function get(url){const r=await fetch(url,{cache:"no-store"});if(!r.ok)throw new Error("Source unavailable");return r.json();}
async function live(){
  if(polling || selection!=="live")return;
  const version=generation;polling=true;
  try {
    const d=await get("/api/live");
    if(version!==generation)return;
    const statuses={running:"Receiving updates",starting:"Starting camera",offline:"No monitor connected",
      ended:"Run ended · final observation",stale:"Connection stale",interrupted:"Run interrupted"};
    const valid=["running","ended"].includes(d.status);
    render(d.layout?d:null,valid?d.sample:null,statuses[d.status]||"Unavailable");$("error").textContent="";
  } catch(e){if(version===generation){render(null,null,"Connection unavailable");$("error").textContent=e.message;}}
  finally{polling=false;}
}
async function selectSource(){
  const version=++generation;selection=$("runs").value;run=null;lastSample=-1;setPlay(false);
  $("error").textContent="";$("playback").hidden=selection==="live";
  if(selection==="live"){render(null,null,"Connecting…");await live();return;}
  try {
    const d=await get("/api/runs/"+encodeURIComponent(selection));if(version!==generation)return;
    if(!d.samples?.length)throw new Error("Replay contains no observations");
    run=d;const end=d.samples.at(-1).t;
    position=selection==="controlled"?Math.min(9.5,end):0;
    $("seek").max=end;drawReplay();
  }catch(e){if(version===generation){render(null,null,"Replay unavailable");$("error").textContent=e.message;}}
}
$("runs").addEventListener("change",selectSource);
$("play").addEventListener("click",()=>{if(!run)return;if(position>=run.samples.at(-1).t)position=0;setPlay(!playing);drawReplay();});
$("seek").addEventListener("input",()=>{setPlay(false);position=Number($("seek").value);drawReplay();});
function tick(now){
  if(playing && run){position=Math.min(run.samples.at(-1).t,position+(now-previousTick)/1000);
    previousTick=now;drawReplay();if(position>=run.samples.at(-1).t)setPlay(false);}
  requestAnimationFrame(tick);
}
(async()=>{try{
  const runs=await get("/api/runs");$("runs").replaceChildren();
  for(const x of [...runs,{id:"live",name:"Live monitor"}]){const o=tag("option",x.name);o.value=x.id;$("runs").append(o);}
  await selectSource();
}catch(e){$("error").textContent=e.message;}
requestAnimationFrame(tick);setInterval(live,1000);})();
