"""Verify 6 candidate product URLs (auth + size via 1-byte range)."""
import requests

COOKIE = "FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;JSESSIONID=1734d4d9ce42e41709fe71bfd2c8;FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;OAuth_Token_Request_State=c32ecd97-864b-4962-beb3-3285f5f1554e;JSESSIONID=17381ce456da5c532a5c922e2a20;FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;"
P = "https://pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/ch2_bundle/cho_bundle/nop"

cands = {
    "OHRC-1": [f"{P}/ohr_collection/data/calibrated/20220914/ch2_ohr_ncp_20220914T0835371412_d_img_d32.zip?ohrc"],
    "OHRC-2": [f"{P}/ohr_collection/data/calibrated/20220914/ch2_ohr_ncp_20220914T1033119094_d_img_d32.zip?ohrc"],
    "TMC2-1": [f"{P}/tmc_collection/data/calibrated/20220221/ch2_tmc_ncf_20220221T1109281684_d_img_d18.zip?tmc2"],
    "TMC2-2": [f"{P}/tmc_collection/data/calibrated/20220403/ch2_tmc_ncf_20220403T1007075726_d_img_d18.zip?tmc2"],
    "IIRS-1": [f"{P}/iir_collection/data/raw/20220221/ch2_iir_nri_20220221T1109265965_d_img_d18.zip?iirs",
               f"{P}/iir_collection/data/calibrated/20220221/ch2_iir_nri_20220221T1109265965_d_img_d18.zip?iirs"],
    "IIRS-2": [f"{P}/iir_collection/data/raw/20220403/ch2_iir_nri_20220403T1007061375_d_img_d18.zip?iirs",
               f"{P}/iir_collection/data/calibrated/20220403/ch2_iir_nri_20220403T1007061375_d_img_d18.zip?iirs"],
}
for tag, urls in cands.items():
    for u in urls:
        try:
            r = requests.get(u, headers={"Cookie": COOKIE, "Range": "bytes=0-0"},
                             stream=True, timeout=(30, 60), allow_redirects=False)
            cr = r.headers.get("Content-Range", "")
            total = cr.split("/")[-1] if "/" in cr else "?"
            print(f"{tag}: HTTP {r.status_code} total={total} via {'raw' if '/raw/' in u else 'calibrated'}")
            r.close()
            if r.status_code in (200, 206):
                break
        except Exception as e:
            print(f"{tag}: ERROR {e}")
