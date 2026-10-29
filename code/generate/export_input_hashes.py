"""Write a local integrity manifest for bundled raw and processed inputs.

The hashes identify the exact packaged bytes. They do not independently verify
that a file matches an upstream provider's record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
INCLUDED_ROOTS = ("raw", "processed", "source_main_external", "yyc200_hyperspectral")


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    files = []
    for folder in INCLUDED_ROOTS:
        for path in sorted((DATA / folder).rglob("*")):
            if path.is_file():
                files.append({
                    "file": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path, "sha256"),
                    "md5": digest(path, "md5"),
                })
    payload = {
        "schema_version": 1,
        "purpose": "Integrity hashes of the exact bundled data bytes; these do not establish upstream-provider identity.",
        "files": files,
    }
    target = DATA / "input_hashes.json"
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)} with {len(files)} file hashes")


if __name__ == "__main__":
    main()
