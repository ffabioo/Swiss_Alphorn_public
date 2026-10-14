"""
Download the published data (Harvard Dataverse, https://doi.org/10.7910/DVN/C1FTYA) into databases/, with the layout
the code expects (utils/paths.py), and check every file against the checksum published by Dataverse.
Plain Python (standard library only), run from the repository root:

    python3 utils/download_data.py              # reduced bases and resonance datasets (14 files, 1.5 GB): all that
                                                #   ./reproduce.sh saved and retrain need besides git
    python3 utils/download_data.py snapshots    # the FE snapshots (35 tar files, 83 GB) into databases/snapshots/,
                                                #   needed only to rebuild the reduced bases (steps 2-3)
    python3 utils/download_data.py --list [snapshots]   # list the files and their sizes, download nothing

Files already present with the right checksum are skipped, so an interrupted download can be resumed.
The trained networks (databases/neural_networks/) are in git and are not downloaded.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.paths import DATABASES  # noqa: E402

SERVER = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/C1FTYA"
FOLDERS = {"data": ("reduced_basis", "resonance_datasets"), "snapshots": ("snapshots",)}
HEADERS = {"User-Agent": "Swiss_Alphorn_public/download_data.py"}   # the server refuses Python's default user agent


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS))


def dataset_files(folders):
    url = f"{SERVER}/api/datasets/:persistentId/versions/:latest-published/files?persistentId={DOI}"
    with get(url) as r:
        files = json.load(r)["data"]
    return [f for f in files if f.get("directoryLabel", "").split("/")[0] in folders]


def checksum(path, kind):
    h = hashlib.new(kind.lower().replace("-", ""))
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(f):
    d = f["dataFile"]
    target = DATABASES / f["directoryLabel"] / d["filename"]
    kind, value = d["checksum"]["type"], d["checksum"]["value"]
    if target.exists() and checksum(target, kind) == value:
        print(f"ok (already there)  {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with get(f"{SERVER}/api/access/datafile/{d['id']}") as r, open(part, "wb") as out:
        for block in iter(lambda: r.read(1 << 24), b""):
            out.write(block)
    if checksum(part, kind) != value:
        part.unlink()
        sys.exit(f"checksum mismatch for {target}: download it again")
    part.rename(target)
    print(f"ok ({d['filesize'] / 1e6:8.1f} MB)  {target}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--list"]
    which = args[0] if args else "data"
    if which not in FOLDERS:
        sys.exit(f"usage: python3 utils/download_data.py [--list] [{' | '.join(FOLDERS)}]")
    files = dataset_files(FOLDERS[which])
    total = sum(f["dataFile"]["filesize"] for f in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB, from {SERVER} ({DOI}) into {DATABASES}/")
    if "--list" in sys.argv[1:]:
        for f in files:
            print(f"{f['dataFile']['filesize'] / 1e6:10.1f} MB  {f['directoryLabel']}/{f['dataFile']['filename']}")
        sys.exit()
    for f in files:
        download(f)
    if which == "snapshots":
        print("unpack them: cd databases/snapshots && for t in snapshots_*.tar; do tar -xf \"$t\"; done && cd -")
