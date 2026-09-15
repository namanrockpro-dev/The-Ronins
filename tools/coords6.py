"""Corner coordinates + centers for the 6 downloaded products."""
import shapefile
from pathlib import Path

BASE = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\shapefiles")
WANT = [
    "ch2_ohr_ncp_20220914T0835371412_d_img_d32",
    "ch2_ohr_ncp_20220914T1033119094_d_img_d32",
    "ch2_tmc_ncf_20220221T1109281684_d_img_d18",
    "ch2_tmc_ncf_20220403T1007075726_d_img_d18",
    "ch2_iir_nri_20220221T1109265965_d_img_d18",
    "ch2_iir_nri_20220403T1007061375_d_img_d18",
]
for shp in [
    BASE / "ohrc" / "ohr_r1_r11_shape_ver4" / "ch2_ohr_cal.shp",
    BASE / "tmc2" / "tmc2_s1_s14_v1_shape" / "ch2_tmc_cal.shp",
    BASE / "iirs" / "iirs_s1_s12_v2_shape" / "ch2_iir_cal.shp",
]:
    r = shapefile.Reader(str(shp))
    fields = [f[0] for f in r.fields[1:]]
    for rec in r.records():
        d = dict(zip(fields, rec))
        if str(d.get("PRODUCT_ID", "")) in WANT:
            ul = (d.get("UL_LAT"), d.get("UL_LON"))
            ur = (d.get("UR_LAT"), d.get("UR_LON"))
            bl = (d.get("BL_LAT"), d.get("BL_LON"))
            br = (d.get("BR_LAT"), d.get("BR_LON"))
            clat = (float(ul[0]) + float(ur[0]) + float(bl[0]) + float(br[0])) / 4
            clon = (float(ul[1]) + float(ur[1]) + float(bl[1]) + float(br[1])) / 4
            print(f"{d['PRODUCT_ID']}  obs={d.get('OBS_ST_TIM')}")
            print(f"  UL=({ul[0]}, {ul[1]})  UR=({ur[0]}, {ur[1]})")
            print(f"  BL=({bl[0]}, {bl[1]})  BR=({br[0]}, {br[1]})")
            print(f"  CENTER=({clat:.4f}, {clon:.4f})")
