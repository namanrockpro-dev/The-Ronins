"""Dump key label fields for all 6 box2 products."""
import re
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\data\raw\box2")
TAGS = ["sun_azimuth", "sun_elevation", "solar_incidence", "emission",
        "phase_angle", "lines", "line_samples", "samples", "data_type",
        "offset", "scaling_factor", "value_offset", "exposure_duration",
        "upper_left_latitude", "upper_left_longitude"]


def get(xml_text, tag):
    m = re.search(rf"<(?:\w+:)?{tag}[^>]*>([^<]+)</", xml_text)
    return m.group(1).strip() if m else "?"


for z in sorted(ROOT.glob("*.zip")):
    with zipfile.ZipFile(z) as a:
        xml_name = next(n for n in a.namelist() if n.endswith(".xml") and "_img_" in n)
        t = a.read(xml_name).decode("utf-8", "ignore")
        img_name = next(n for n in a.namelist()
                        if n.endswith((".img", ".qub", ".dat")))
        isize = a.getinfo(img_name).file_size
    print(f"== {z.name}  img_bytes={isize}")
    print("   " + " | ".join(f"{k}={get(t, k)}" for k in TAGS))
