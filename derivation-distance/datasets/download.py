#!/usr/bin/env python3
"""Fetch the benchmark data (23 gzipped CSVs, ~70 MB) into datasets/<ds>/data/.

The data is distributed as a single tarball, separate from the git repo:

    python datasets/download.py            # fetch + verify (skips if already valid)
    python datasets/download.py --check    # verify existing files only, no network
    python datasets/download.py --url URL  # fetch from an alternate location

The archive URL can also be overridden with the DERIVATION_DISTANCE_DATA_URL
environment variable. Every extracted file is verified against
datasets/checksums.json (SHA-256); the script fails loudly on any mismatch.

Uses only the Python standard library. If the default URL returns 404 the
release may still be private — authenticate and use:
    gh release download derivation-distance-data-v1 -R getcassis/research \
        -O /tmp/dd-data.tar.gz && python datasets/download.py --url file:///tmp/dd-data.tar.gz
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_URL = (
    "https://github.com/getcassis/research/releases/download/"
    "derivation-distance-data-v1/derivation-distance-data.tar.gz"
)


def load_checksums():
    with open(os.path.join(HERE, "checksums.json")) as f:
        return json.load(f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(sums, verbose=True):
    """Returns (ok, missing, bad)."""
    missing, bad = [], []
    for rel, want in sorted(sums.items()):
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            missing.append(rel)
        elif sha256(p) != want:
            bad.append(rel)
    ok = not missing and not bad
    if verbose:
        n = len(sums)
        print(f"verified {n - len(missing) - len(bad)}/{n} files"
              + (f", missing {len(missing)}" if missing else "")
              + (f", CHECKSUM MISMATCH {len(bad)}: {bad}" if bad else ""))
    return ok, missing, bad


def fetch(url):
    print(f"downloading {url}")
    fd, tmp = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
        return tmp
    except Exception:
        os.unlink(tmp)
        raise


def extract(archive, sums):
    """Extract only the expected members, into datasets/."""
    expected = set(sums)
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name in expected]
        if len(members) != len(expected):
            found = {m.name for m in members}
            raise SystemExit(f"archive is missing files: {sorted(expected - found)}")
        for m in members:
            tar.extract(m, HERE, filter="data")
    print(f"extracted {len(members)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("DERIVATION_DISTANCE_DATA_URL", DEFAULT_URL))
    ap.add_argument("--check", action="store_true", help="verify existing files only")
    ap.add_argument("--force", action="store_true", help="re-download even if files verify")
    args = ap.parse_args()

    sums = load_checksums()
    ok, _, bad = verify(sums, verbose=True)
    if args.check:
        sys.exit(0 if ok else 1)
    if ok and not args.force:
        print("all files present and verified; nothing to do")
        return
    if bad:
        raise SystemExit("existing files fail checksum verification; refusing to overwrite "
                         f"automatically: {bad}\nDelete them and re-run to re-download.")

    tmp = fetch(args.url)
    try:
        extract(tmp, sums)
    finally:
        os.unlink(tmp)
    ok, missing, bad = verify(sums, verbose=True)
    if not ok:
        raise SystemExit(f"verification failed after download: missing={missing} bad={bad}")
    print("done")


if __name__ == "__main__":
    main()
