"""Download the viral-db.csv dataset (~60 MB) from Google Drive into data/."""

from pathlib import Path

import gdown

FILE_ID = "15x7NVNCivXGwS9OpuJfa4hm0FnP3lm4c"
DEST = Path(__file__).resolve().parent.parent / "data" / "viral-db.csv"

if __name__ == "__main__":
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"{DEST} already exists - skipping download.")
    else:
        gdown.download(id=FILE_ID, output=str(DEST), quiet=False)
