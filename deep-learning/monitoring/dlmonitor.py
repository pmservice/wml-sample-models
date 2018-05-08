# Simple utility for monitoring deep learning jobs

import sys
import json
import os
from base64 import b64encode
import ssl

try:
    from lomond import WebSocket
except:
    print("This program depends on the lomond python library.  Please use \"pip install lomond\" to install it and try again.")
    sys.exit(-1)

if len(sys.argv) != 2:
    print("Usage: python dlmonitor.py training-run-id")
    sys.exit(-1)

model_id=sys.argv[1]

if "ML_ENV" not in os.environ or "ML_USERNAME" not in os.environ or "ML_PASSWORD" not in os.environ:
    print("Please ensure environment variables ML_USERNAME, ML_PASSOWRD and ML_ENV are defined")
    print("These are the same environment variables needed for the machine learning CLI.  Please define them and try again.")
    sys.exit(-1)

ml_url=os.environ["ML_ENV"]
ml_username = os.environ["ML_USERNAME"]
ml_password = os.environ["ML_PASSWORD"]

# handle python2/python3 compatibility

isPy2 = sys.version.startswith("2")

if isPy2:
    def toBytes(val,enc):
        return bytes(val)
    def toStr(val,enc):
        return str(val)
    import httplib
    from urlparse import urlparse

else:
    def toBytes(val,enc):
        return bytes(val,enc)
    def toStr(val,enc):
        return str(val,enc)
    import http.client as httplib
    from urllib.parse import urlparse

# get a WML token

token_host = urlparse(ml_url).netloc
conn = httplib.HTTPSConnection(token_host, context=ssl._create_unverified_context())
userAndPass = b64encode(toBytes(ml_username + ":" + ml_password, "utf-8")).decode("ascii")
headers = {'Authorization': 'Basic %s' % userAndPass}
conn.request('GET', "/v2/identity/token", headers=headers)
resp = conn.getresponse()

if resp.status == 200:
    token = json.loads(toStr(resp.read(),"utf-8"))["token"]
else:
    print("Error %d obtaining token"%(resp.status))
    sys.exit(-1)

try:
    websocket = WebSocket(ml_url.replace("https","wss").replace("http","ws")+"/v3/models/"+model_id+"/monitor?last=10")
    websocket.add_header(toBytes("Authorization","utf-8"),toBytes("bearer "+token,"utf-8"))
    for event in websocket:
        if event.name == 'text':
            status = json.loads(event.text)
            if "status" in status:
                status = status["status"]
                if "message" in status:
                    message = status["message"]
                    sys.stdout.write(message)
                    sys.stdout.flush()
except Exception as ex:
    print("ERROR:"+ex)