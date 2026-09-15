"""Find true same-coordinate triplets: OHRC strip overlapped by TMC-2 AND IIRS."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\shapefiles")


def load(shp):
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    recs = []
    for shp_obj, rec in zip(r.shapes(), r.records()):
        d = dict(zip(fields, rec))
        try:
            x0, y0, x1, y1 = shp_obj.bbox
        except Exception:
            continue
        lon0, lon1 = min(x0, x1), max(x0, x1)
        lat0, lat1 = min(y0, y1), max(y0, y1)
        if lon0 > 180:
            lon0 -= 360
            lon1 -= 360
        recs.append({"attrs": d, "bbox": (lat0, lat1, lon0, lon1)})
    return recs


def overlaps(a, b):
    return (a[2] <= b[3] and a[3] >= b[2] and a[0] <= b[1] and a[1] >= b[0])


ohrc = load(BASE / "ohrc" / "ohr_r1_r11_shape_ver4" / "ch2_ohr_cal.shp")
tmc = load(BASE / "tmc2" / "tmc2_s1_s14_v1_shape" / "ch2_tmc_cal.shp")
iirs = load(BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp")
print(f"OHRC={len(ohrc)} TMC2={len(tmc)} IIRS={len(iirs)}")

triplets = []
for o in ohrc:
    ob = o["bbox"]
    tm = [t for t in tmc if overlaps(ob, t["bbox"])]
    ii = [t for t in iirs if overlaps(ob, t["bbox"])]
    if tm and ii:
        triplets.append((o, tm[0], ii[0], len(tm), len(ii)))

print(f"\nOHRC strips with BOTH TMC-2 and IIRS overlap: {len(triplets)}")
for o, t, i, nt, ni in triplets[:15]:
    oa = o["attrs"]
    print("-" * 100)
    print(f"OHRC {oa.get('PRODUCT_ID')} obs={oa.get('OBS_ST_TIM')} "
          f"bbox(lat {o['bbox'][0]:.3f}..{o['bbox'][1]:.3f}, lon {o['bbox'][2]:.3f}..{o['bbox'][3]:.3f})")
    print(f"   DOWNLOAD={oa.get('DOWNLOAD')} BROWSE={oa.get('BROWSE')}")
    print(f"   TMC2 x{nt} e.g. {t['attrs'].get('PRODUCT_ID')} "
          f"bbox(lat {t['bbox'][0]:.2f}..{t['bbox'][1]:.2f}, lon {t['bbox'][2]:.2f}..{t['bbox'][3]:.2f})")
    print(f"   TMC2 DOWNLOAD={t['attrs'].get('DOWNLOAD')}")
    print(f"   IIRS x{ni} e.g. {i['attrs'].get('PRODUCT_ID')} "
          f"bbox(lat {i['bbox'][0]:.2f}..{i['bbox'][1]:.2f}, lon {i['bbox'][2]:.2f}..{i['bbox'][3]:.2f})")
    print(f"   IIRS DOWNLOAD={i['attrs'].get('DOWNLOAD')}")
