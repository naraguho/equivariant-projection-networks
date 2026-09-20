#!/usr/bin/env python3
"""Download and verify the real manuscript data from the GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/naraguho/equivariant-projection-networks/releases/download/data-v1"
ASSETS = {
    "fk_edkmc_rc10_full.csv.gz": "efef9ace24ac7c364a810733ad6eba21afa937de095c87c2841689e1c6502643",
    "fk_heldout_ed_correlation_reference.npz": "15307ddabd3c098bc9fbf2078a12a5007a5c42c306d7ef0a3ae8e3947ea99774",
    "holstein_ed_forces_full.zip": "9d986a7555e39c9042a0fca44f193be638b6c04d5d5a1699d80ecde7bd63d3f3",
    "manuscript_benchmarks.zip": "51d32bd4c20d30bba6a9a6bad795a2e2390bf9babea105900840d142f739e3e0",
    "trained_checkpoints.zip": "b93439add4b1d3ed16cfe284c24bdaac719e00e61ab1e480108e5c5d180a2560",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*", choices=ASSETS, default=list(ASSETS))
    parser.add_argument("--output", type=Path, default=Path("data/full"))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for name in args.assets:
        destination = args.output / name
        if not destination.exists() or sha256(destination) != ASSETS[name]:
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{RELEASE}/{name}", destination)
        actual = sha256(destination)
        if actual != ASSETS[name]:
            raise RuntimeError(f"checksum mismatch for {name}: {actual}")
        print(f"verified {name}")
        if args.extract and destination.suffix == ".zip":
            target = args.output / destination.stem
            with zipfile.ZipFile(destination) as archive:
                archive.extractall(target)
            print(f"extracted to {target}")


if __name__ == "__main__":
    main()

