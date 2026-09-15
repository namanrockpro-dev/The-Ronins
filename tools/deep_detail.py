"""Deep detail: all XML tags, geometry CSV structure, hdr, readme per sensor."""
import re
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman\data\raw\box2")
REPS = {
    "OHRC": "ch2_ohr_ncp_20220914T0835371412_d_img_d32.zip",
    "TMC2": "ch2_tmc_ncf_20220221T1109281684_d_img_d18.zip",
    "IIRS": "ch2_iir_nri_20220221T1109265965_d_img_d18.zip",
}
for sensor, zname in REPS.items():
    print("#" * 100)
    print(f"SENSOR: {sensor}  ({zname})")
    with zipfile.ZipFile(ROOT / zname) as a:
        names = a.namelist()
        xml = next(n for n in names if n.endswith(".xml") and "_img_" in n)
        t = a.read(xml).decode("utf-8", "ignore")
        # every unique element + one sample value
        tags = {}
        for m in re.finditer(r"<(?:\w+:)?([\w]+)([^>]*)>([^<]{1,80})</", t):
            tag, attrs, val = m.group(1), m.group(2).strip(), m.group(3).strip()
            if tag not in tags:
                tags[tag] = (attrs[:60], val[:60])
        print(f"  label {xml} ({len(t)} chars), {len(tags)} unique elements:")
        for tag, (attrs, val) in sorted(tags.items()):
            print(f"    {tag}  [{attrs}] = {val}")
        # geometry csv
        gcsv = next((n for n in names if n.endswith(".csv")), None)
        if gcsv:
            raw = a.read(gcsv).decode("utf-8", "ignore").splitlines()
            print(f"  geometry {gcsv}: {len(raw)} lines")
            for line in raw[:3]:
                print(f"    {line[:200]}")
        hdr = next((n for n in names if n.endswith(".hdr")), None)
        if hdr:
            print(f"  hdr: {a.read(hdr).decode('utf-8','ignore')!r}")
        print("  readme:", a.read("miscellaneous/readme.txt").decode("utf-8", "ignore")[:400])
