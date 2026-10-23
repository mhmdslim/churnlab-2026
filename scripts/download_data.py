"""Download the IBM Telco Customer Churn CSV to data/raw/telco.csv."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

COMMIT = "1fd6fd70906479be1712e64cfc0ea89f58466629"
URL = (
    f"https://raw.githubusercontent.com/IBM/"
    f"telco-customer-churn-on-icp4d/{COMMIT}/"
    f"data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = (
    "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
)

dest = Path("data/raw/telco.csv")
dest.parent.mkdir(parents=True, exist_ok=True)

print(f"Downloading from {URL} ...")
data = urlopen(URL).read()
sha = hashlib.sha256(data).hexdigest()

if sha != EXPECTED_SHA256:
    raise SystemExit(f"SHA-256 mismatch!\n  got:      {sha}\n  expected: {EXPECTED_SHA256}")

dest.write_bytes(data)
print(f"Saved to {dest}  ({len(data):,} bytes)")
print(f"SHA-256: {sha}")

meta = {
    "url": URL,
    "commit": COMMIT,
    "sha256": sha,
    "retrieved": datetime.now(timezone.utc).isoformat(),
}
meta_path = Path("data/raw/telco.source.json")
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(f"Source metadata: {meta_path}")
