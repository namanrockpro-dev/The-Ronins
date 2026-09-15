"""Full inventory: every file in each box2 zip + all key label fields."""
import re
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\data\raw\box2")


def tag(xml, t):
    m = re.search(rf"<(?:\w+:)?{t}[^>]*>([^<]+)</", xml)
    return m.group(1).strip() if m else "—"


for z in sorted(ROOT.glob("*.zip")):
    print("=" * 100)
    print(f"ZIP: {z.name}  ({z.stat().st_size:,} bytes)")
    with zipfile.ZipFile(z) as a:
        for n in sorted(a.namelist()):
            info = a.getinfo(n)
            if n.endswith("/"):
                continue
            print(f"   {info.file_size:>12,}  {n}")
        xml_name = next(n for n in a.namelist() if n.endswith(".xml") and "_img_" in n)
        t = a.read(xml_name).decode("utf-8", "ignore")
    print("  -- label --")
    for k in ["logical_identifier", "title", "product_class", "start_date_time",
              "stop_date_time", "exposure_duration", "data_type",
              "sun_azimuth", "sun_elevation", "solar_incidence",
              "upper_left_latitude", "upper_left_longitude",
              "upper_right_latitude", "upper_right_longitude",
              "lower_left_latitude", "lower_left_longitude",
              "lower_right_latitude", "lower_right_longitude"]:
        print(f"   {k} = {tag(t, k)}")
    axes = re.findall(r"<axis_name>([^<]+)</axis_name>.*?<elements>(\d+)</elements>", t, re.S)
    print("   axes =", axes)
