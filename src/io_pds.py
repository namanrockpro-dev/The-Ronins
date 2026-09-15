"""PDS4 I/O for Chandrayaan-2 OHRC / TMC-2 / IIRS products inside PRADAN zips.

Reads directly from zip (no full extract): labels parsed for geometry +
sun angles, pixel data streamed via a disk cache + np.memmap.
"""
import re
import zipfile
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _find(zip_path, *exts):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    for ext in exts:
        for n in names:
            if n.endswith(ext) and "_img_" in n:
                return n
    raise FileNotFoundError(f"no image payload in {zip_path}")


def read_label(zip_path):
    """Return (label_xml_text, payload_name_inside_zip)."""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        xml = next(n for n in names if n.endswith(".xml") and "_img_" in n)
        pay = next(n for n in names if n.endswith((".img", ".qub", ".dat")))
        return z.read(xml).decode("utf-8", "ignore"), pay


def _tag(xml, tag):
    m = re.search(rf"<(?:\w+:)?{tag}[^>]*>([^<]+)</", xml)
    return m.group(1).strip() if m else None


def parse_meta(zip_path):
    """Metadata dict: sensor, dims, dtype, sun angles, corner lat/lon."""
    xml, pay = read_label(zip_path)
    name = Path(zip_path).name
    sensor = "OHRC" if "_ohr_" in name else ("TMC-2" if "_tmc_" in name else "IIRS")
    axes = re.findall(r"<axis_name>([^<]+)</axis_name>.*?<elements>(\d+)</elements>",
                      xml, re.S)
    dims = {a.strip().upper(): int(e) for a, e in axes}
    dtype = (_tag(xml, "data_type") or "").lower()
    corners = {}
    for c in ("upper_left", "upper_right", "lower_left", "lower_right"):
        la = _tag(xml, f"{c}_latitude")
        lo = _tag(xml, f"{c}_longitude")
        corners[c] = (float(la), float(lo)) if la and lo else None
    f = lambda k: float(_tag(xml, k)) if _tag(xml, k) not in (None, "?") else None
    try:
        dt = {"unsignedbyte": np.uint8, "unsignedlsb2": "<u2",
              "unsignedmsb2": ">u2"}.get(dtype.replace(" ", ""), np.uint8)
    except Exception:
        dt = np.uint8
    return {"sensor": sensor, "file": name, "payload": pay, "dims": dims,
            "dtype": str(np.dtype(dt)), "dtype_raw": dtype,
            "sun_azimuth": f("sun_azimuth"), "sun_elevation": f("sun_elevation"),
            "incidence": f("solar_incidence"), "corners": corners,
            "obs_start": _tag(xml, "start_date_time")}


def _extract(zip_path, pay):
    dest = CACHE / f"{Path(zip_path).stem}__{Path(pay).name}"
    if not dest.exists():
        with zipfile.ZipFile(zip_path) as z:
            with z.open(pay) as src, open(dest, "wb") as dst:
                while True:
                    b = src.read(1 << 24)
                    if not b:
                        break
                    dst.write(b)
    return dest


def load_array(zip_path, band=None):
    """Memmapped array + meta. IIRS: band index or mean of selected bands."""
    meta = parse_meta(zip_path)
    dest = _extract(zip_path, meta["payload"])
    dt = np.dtype(meta["dtype"])
    if meta["sensor"] == "IIRS":
        d = meta["dims"]
        nb, nl, ns = d.get("BAND", 256), d.get("LINE", 1), d.get("SAMPLE", 1)
        arr = np.memmap(dest, dtype=dt, mode="r", shape=(nb, nl, ns))
        if band is None:  # near-IR-ish mean excluding noisy edges
            b0, b1 = max(0, nb // 8), min(nb, nb * 7 // 8)
            img = np.asarray(arr[b0:b1].mean(axis=0), dtype=np.float32)
        else:
            img = np.asarray(arr[int(band)], dtype=np.float32)
        return img, meta
    d = meta["dims"]
    nl, ns = d.get("LINE", 0), d.get("SAMPLE", d.get("LINE_SAMPLES", 0))
    arr = np.memmap(dest, dtype=dt, mode="r", shape=(nl, ns))
    return np.asarray(arr, dtype=np.float32), meta


def load_preview(zip_path, max_side=1600):
    """Float32 preview (longest side <= max_side) + meta + scale factor.

    Uses strided memmap reads so 900MB+ strips never fully materialize.
    """
    import cv2
    meta = parse_meta(zip_path)
    dest = _extract(zip_path, meta["payload"])
    dt = np.dtype(meta["dtype"])
    d = meta["dims"]
    if meta["sensor"] == "IIRS":
        nb, nl, ns = d.get("BAND", 256), d.get("LINE", 1), d.get("SAMPLE", 1)
        step = max(1, int(max(nl, ns) / max_side))
        mm = np.memmap(dest, dtype=dt, mode="r", shape=(nb, nl, ns))
        b0, b1 = max(0, nb // 8), min(nb, nb * 7 // 8)
        img = np.asarray(mm[b0:b1, ::step, ::step].mean(axis=0), dtype=np.float32)
    else:
        nl, ns = d.get("LINE", 0), d.get("SAMPLE", d.get("LINE_SAMPLES", 0))
        step = max(1, int(max(nl, ns) / max_side))
        mm = np.memmap(dest, dtype=dt, mode="r", shape=(nl, ns))
        img = np.asarray(mm[::step, ::step], dtype=np.float32)
    h, w = img.shape
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    full_scale = (img.shape[1] / ns) if ns else 1.0
    return img, meta, full_scale
