"""List all OHRC/TMC2/IIRS strips overlapping box2 (lat 4.5-5.7, lon -125.8--125.3)."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\shapefiles")
BOX = (4.5, 5.7, -125.8, -125.3)  # lat0, lat1, lon0, lon1


def load(shp):
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    out = []
    for s, rec in zip(r.shapes(), r.records()):
        d = dict(zip(fields, rec))
        try:
            x0, y0, x1, y1 = s.bbox
        except Exception:
            continue
        lon0, lon1 = min(x0, x1), max(x0, x1)
        lat0, lat1 = min(y0, y1), max(y0, y1)
        if lon0 > 180:
            lon0 -= 360
            lon1 -= 360
        if lon0 <= BOX[3] and lon1 >= BOX[2] and lat0 <= BOX[1] and lat1 >= BOX[0]:
            out.append((d, (lat0, lat1, lon0, lon1)))
    return out


for name, p in [
    ("OHRC", BASE / "ohrc" / "ohr_r1_r11_shape_ver4" / "ch2_ohr_cal.shp"),
    ("TMC2", BASE / "tmc2" / "tmc2_s1_s14_v1_shape" / "ch2_tmc_cal.shp"),
    ("IIRS", BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp"),
]:
    hits = load(p)
    print(f"===== {name}: {len(hits)} overlapping =====")
    for d, bb in hits[:12]:
        print(f"{d.get('PRODUCT_ID')} | obs={d.get('OBS_ST_TIM')} | "
              f"lat {bb[0]:.2f}..{bb[1]:.2f} lon {bb[2]:.2f}..{bb[3]:.2f} | DL={d.get('DOWNLOAD')}")
