"""Render a names-free vector plan in the calibrated camera coordinate system."""
from xml.etree.ElementTree import Element, SubElement, tostring

SEAT_COLORS = {"available": "#b9e77c", "occupied": "#8dbad5", "uncertain": "url(#hatch)"}


def camera_plan(layout, door, sample):
    width = 760
    height = width / layout.get("aspect_ratio", 2)
    svg = Element("svg", xmlns="http://www.w3.org/2000/svg", viewBox=f"0 0 {width} {height}")
    def draw(parent, kind, **attrs):
        return SubElement(parent, kind, {k.replace("_", "-"): str(v) for k, v in attrs.items()})

    def pixel(point):
        return (width*point[0], height*point[1])

    def polygon(coords, **style):
        points = " ".join(f"{x:.2f},{y:.2f}" for x,y in map(pixel, coords))
        draw(svg, "polygon", points=points, **style)

    def label(x, y, text, size=13, color="#e3efe1", align="start"):
        draw(svg, "text", x=x, y=y, font_family="monospace", font_size=size, fill=color,
             font_weight=600, text_anchor=align).text = text

    defs = draw(svg, "defs")
    dots = draw(defs, "pattern", id="grid", width=16, height=16, patternUnits="userSpaceOnUse")
    draw(dots, "circle", cx=1, cy=1, r=0.85, fill="#477065")
    hatch = draw(defs, "pattern", id="hatch", width=7, height=7, patternUnits="userSpaceOnUse")
    draw(hatch, "rect", width=7, height=7, fill="#526260")
    draw(hatch, "path", d="M-1,1 l2,-2 M0,7 l7,-7 M6,8 l2,-2", stroke="#a0aeab")
    draw(svg, "rect", width=width, height=height, rx=12, fill="#112e2b")
    draw(svg, "rect", width=width, height=height, rx=12, fill="url(#grid)")
    for name, coords in layout.get("zones", {}).items():
        if coords:
            polygon(coords, fill={"front":"#294e43","middle":"#23463e","rear":"#203e38"}.get(name,"#203e38"),
                    stroke="#749286", stroke_width=1.4)
            x,y = pixel(coords[0])
            label(x+8,y+17,name.upper())
    for name, coords in layout.get("passages",{}).items():
        if coords:
            alert = sample.get("passages",{}).get(name,{}).get("status") == "review"
            polygon(coords, fill="#d78737" if alert else "none", fill_opacity=.38,
                    stroke="#ffb46b" if alert else "#80a395",stroke_width=1.8,stroke_dasharray="6 5")
            if alert:
                x = sum(p[0] for p in coords)*width/len(coords)
                y = sum(p[1] for p in coords)*height/len(coords)
                label(x,y+4,name.upper()+" · REVIEW",12,"#ffe4bb","middle")
    for name, rect in layout.get("seats",{}).items():
        x,y = pixel(rect[:2])
        w,h = pixel((rect[2]-rect[0],rect[3]-rect[1]))
        fill = SEAT_COLORS.get(sample.get("seats",{}).get(name), "url(#hatch)")
        draw(svg,"rect",x=x,y=y,width=w,height=h,rx=min(7,w/6),fill=fill,stroke="#183e35",stroke_width=2)
        draw(svg,"path",d=f"M{x+w*.14},{y+h*.80}h{w*.72}",stroke="#345b50",stroke_width=1.5)
        if w>18 and h>14:
            label(x+w/2,y+h*.50+4,name,min(14,w*.42),"#102e29" if fill!="url(#hatch)" else "#ffffff","middle")
    if door:
        a,b = pixel(door["start"]),pixel(door["end"])
        draw(svg,"line",x1=a[0],y1=a[1],x2=b[0],y2=b[1],stroke="#ffba71",stroke_width=4)
        for x,y in (a,b):
            draw(svg,"circle",cx=x,cy=y,r=4,fill="#ffcf99")
        if not layout.get("seats"):
            x,y = pixel(door["inside"])
            label(x,y,"INTERIOR",17,align="middle")
    return tostring(svg, encoding="unicode")
