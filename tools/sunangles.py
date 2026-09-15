"""Sun/illumination geometry for the 6 box2 products (from footprint attributes)."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\shapefiles")
WANT = {
    "ch2_ohr_ncp_20220914T0835371412_d_img_d32",
    "ch2_ohr_ncp_20220914T1033119094_d_img_d32",
    "ch2_tmc_ncf_20220221T1109281684_d_img_d18",
    "ch2_tmc_ncf_20220403T1007075726_d_img_d18",
    "ch2_iir_nri_20220221T1109265965_d_img_d18",
    "ch2_iir_nri_20220403T1007061375_d_img_d18",
}
for shp in [
    BASE / "ohrc" / "ohr_r1_r11_shape_ver4" / "ch2_ohr_cal.shp",
    BASE / "tmc2" / "tmc2_s1_s14_v1_shape" / "ch2_tmc_cal.shp",
    BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp",
]:
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    for rec in r.records():
        d = dict(zip(fields, rec))
        pid = str(d.get("PRODUCT_ID", ""))
        if pid in WANT:
            print(f"{pid}")
            print(f"   obs={d.get('OBS_ST_TIM')}  inc={d.get('INC_ANGLE')}  "
                  f"emi={d.get('EMI_ANGLE')}  pha={d.get('PHA_ANGLE')}")
