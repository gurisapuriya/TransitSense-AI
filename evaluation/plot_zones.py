"""Render the spatial evaluation figure from independent counts and saved estimates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def plot(output):
    records = json.loads((ROOT / "evaluation/cabin_reference.json").read_text())
    scenes = [r for r in records if r["phase"] == "calibration"]
    zones = ("front", "middle", "rear")
    fig, axes = plt.subplots(1,len(scenes), figsize=(9,3.4), sharey=True)
    for ax, ref in zip(axes,scenes):
        replay = ROOT / "dashboard/replays" / (ref["id"] + ".json")
        observed = json.loads(replay.read_text())["samples"][-1]["zones"]
        xs = range(3)
        left = ax.bar([x-.18 for x in xs], [ref["visible_people"][z] for z in zones],
                      .34,color="#233b52",label="Reviewed")
        right = ax.bar([x+.18 for x in xs], [observed[z]["people"] for z in zones],
                       .34,color="#339989",label="Estimated")
        ax.bar_label(left,padding=2,fontsize=9)
        ax.bar_label(right,padding=2,fontsize=9)
        ax.set_xticks(list(xs), [z.title() for z in zones])
        ax.set_title(ref["id"].title()+" cabin",loc="left",fontweight="bold")
        ax.set_ylim(0,5)
        ax.set_yticks(range(5))
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Visible people")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=2,frameon=False)
    fig.tight_layout(rect=[0,0,1,.87])
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"docs/zone-evaluation.png")
    plot(parser.parse_args().output)
