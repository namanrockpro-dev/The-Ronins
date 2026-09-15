"""Intersect OHRC/IIRS/TMC2 footprint shapefiles with target bbox."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\shapefiles")
# TMC-2 nadir 20260811 footprint
TGT = {"lat_min": 18.9, "lat_max": 21.6, "lon_min": 162.3, "lon_max": 163.3}


def show_prj(path):
    for f in sorted(Path(path).rglob("*.prj")):
        print(f.name, "->", f.read_text(errors="ignore").strip()[:160])


def search(shp_path, latlon_fields=("lat", "lon")):
    r = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in r.fields[1:]]
    print(f"\n== {shp_path.name}: {len(r)} records, fields={fields}")
    hits = []
    for i, (shp, rec) in enumerate(zip(r.shapes(), r.records())):
        try:
            x0, y0, x1, y1 = shp.bbox  # lon_min, lat_min, lon_max, lat_max (if geo)
        except Exception:
            continue
        lon0, lon1, lat0, lat1 = min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)
        # handle 0..360 vs -180..180
        if lon0 > 180:
            lon0 -= 360
            lon1 -= 360
        if (lon0 <= TGT["lon_max"] and lon1 >= TGT["lon_min"]
                and lat0 <= TGT["lat_max"] and lat1 >= TGT["lat_min"]):
            hits.append((dict(zip(fields, rec)), (lat0, lat1, lon0, lon1)))
    print(f"   INTERSECTING records: {len(hits)}")
    for rec, bb in hits[:20]:
        name = rec.get("Name") or rec.get("FILE_NAME") or rec.get("FileName") or rec.get("PRODUCT") or str(rec)[:120]
        print(f"   - {name} bbox(lat {bb[0]:.3f}..{bb[1]:.3f}, lon {bb[2]:.3f}..{bb[3]:.3f})")
    return hits


print("--- projections ---")
show_prj(BASE / "ohrc")
print("--- OHRC ---")
search(BASE / "ohrc" / "ohr_r1_r11_shape_ver4" / "ch2_ohr_cal.shp")
print("--- IIRS ---")
search(BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp")
