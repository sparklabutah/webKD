"""Install WebKD integration files into the documented LLaMA-Factory checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="LLaMA-Factory checkout")
    args = parser.parse_args()
    overlay = Path(__file__).resolve().parents[1] / "integrations" / "llamafactory"
    manifest = json.loads((overlay / "base.json").read_text())
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["revision"]:
        parser.error(f"Checkout must be at revision {manifest['revision']}")
    target_root = args.checkout / "src" / "llamafactory"
    for name, digest in manifest["files"].items():
        target = target_root / name
        source = overlay / name
        if target.read_bytes() == source.read_bytes():
            continue
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            parser.error(f"Refusing to overwrite modified upstream file: {name}")
    for name in manifest["files"]:
        shutil.copy2(overlay / name, target_root / name)
    print("Installed WebKD preprocessing, collation, and training integration.")


if __name__ == "__main__":
    main()
