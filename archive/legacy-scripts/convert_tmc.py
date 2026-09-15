"""Convert 6x Chandrayaan-2 TMC-2 PDS4 .img (UnsignedLSB2) inside zips to full-res 8-bit PNGs."""
import zipfile, xml.etree.ElementTree as ET, pathlib, numpy as np, cv2, os

BASE = pathlib.Path(r"C:\Users\swati\OneDrive\Desktop\TestNaman")
OUT = BASE / "converted_fullres"
OUT.mkdir(exist_ok=True)

def get_dims_and_imgname(zip_path):
    z = zipfile.ZipFile(zip_path)
    xmln = [n for n in z.namelist() if n.endswith(".xml") and "_d_img_" in n][0]
    root = ET.fromstring(z.read(xmln))
    NS = "{http://pds.nasa.gov/pds4/pds/v1}"
    axes = root.findall(f".//{NS}Axis_Array")
    lines = int(axes[0].find(f"{NS}elements").text)
    samples = int(axes[1].find(f"{NS}elements").text)
    # file_name tag
    fn = root.find(f".//{NS}File/{NS}file_name")
    imgname = fn.text.strip() if fn is not None else None
    # find actual .img entry (name may differ in path)
    imgn = [n for n in z.namelist() if n.endswith(".img")][0]
    z.close()
    return lines, samples, imgn

for zp in sorted(BASE.glob("ch2_tmc_*.zip")):
    short = zp.stem  # e.g. ch2_tmc_ncn_20260811T1856504555_d_img_d18
    out_png = OUT / (short + "_full.png")
    if out_png.exists():
        print(f"SKIP exists: {out_png.name} ({out_png.stat().st_size/1e6:.1f} MB)")
        continue
    lines, samples, imgn = get_dims_and_imgname(zp)
    print(f"\n{zp.name}: {lines} x {samples} -> {imgn}")
    print("  reading raw uint16 from zip (no full extract)...")
    z = zipfile.ZipFile(zp)
    raw = z.read(imgn)
    z.close()
    expected = lines * samples * 2
    print(f"  bytes: {len(raw)/1e6:.1f} MB (expected {expected/1e6:.1f} MB)")
    arr = np.frombuffer(raw, dtype="<u2").reshape(lines, samples)
    # percentile stretch 2-98% to 8-bit (same as Module 2 philosophy, full-res preserve)
    p2, p98 = np.percentile(arr, (2, 98))
    print(f"  raw range [{arr.min()}..{arr.max()}] p2={p2:.1f} p98={p98:.1f}")
    if p98 > p2:
        norm = np.clip((arr.astype(np.float32) - p2) / (p98 - p2) * 255.0, 0, 255).astype(np.uint8)
    else:
        norm = ((arr >> 8)).astype(np.uint8)
    del arr, raw
    print(f"  writing {out_png.name} ...")
    cv2.imwrite(str(out_png), norm)
    print(f"  DONE {out_png.stat().st_size/1e6:.1f} MB")
    del norm

print("\nAll conversions done:")
for p in sorted(OUT.glob("*.png")):
    print(f" {p.name} {p.stat().st_size/1e6:.1f} MB")
