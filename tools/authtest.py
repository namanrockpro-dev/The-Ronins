"""Quick auth test: 1KB range request with PRADAN session cookie."""
import requests

cookie_string = "FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;JSESSIONID=1734d4d9ce42e41709fe71bfd2c8;FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;OAuth_Token_Request_State=c32ecd97-864b-4962-beb3-3285f5f1554e;JSESSIONID=17381ce456da5c532a5c922e2a20;FGTServer=5DB1E9B68132028CF7976EE4DF4CBB47C2F908C978D8DADB79380837E680FA20672DD56B0798AF391BF6;"
url = ("https://pradan.issdc.gov.in/ch2/protected/downloadData/POST_OD/isda_archive/"
       "ch2_bundle/cho_bundle/nop/ohr_collection/data/calibrated/20231004/"
       "ch2_ohr_ncp_20231004T0406038822_d_img_d18.zip?ohrc")
h = {"Cookie": cookie_string, "Range": "bytes=0-1023"}
r = requests.get(url, headers=h, stream=True, timeout=(30, 60), allow_redirects=False)
print("status:", r.status_code)
print("content-length:", r.headers.get("Content-Length"), "| accept-ranges:", r.headers.get("Accept-Ranges"))
print("location:", r.headers.get("Location"))
body = r.content[:80]
print("first bytes:", body[:40])
