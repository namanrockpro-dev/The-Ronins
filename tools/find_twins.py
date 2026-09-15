"""Find raw-TMC2 + calibrated-IIRS twins of our two strips."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\data\shapefiles")
WANT_T = ["20220221T1109", "20220403T1007"]


def scan(shp, tag):
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    for rec in r.records():
        d = dict(zip(fields, rec))
        pid = str(d.get("PRODUCT_ID", ""))
        if any(w in pid for w in WANT_T):
            print(f"{tag}: {pid} | DL={d.get('DOWNLOAD')}")


scan(BASE / "tmc2" / "tmc2_s1_s14_v1_shape" / "ch2_tmc_raw.shp", "TMC2-RAW")
scan(BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp", "IIRS-CAL")
