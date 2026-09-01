#!/usr/bin/env python3
"""Validate inert Bundle distributions without importing vLLM."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path

EXPECTED_DISTRIBUTION = "vllm-ascend-adaptive-quantized-kv-hust"
EXPECTED_VERSION = "0.1.0.dev0"
EXPECTED_BUNDLE_ID = "org.vllm-hust.ascend-adaptive-quantized-kv"
EXPECTED_ENTRY_POINT = "vllm_ascend_adaptive_quantized_kv"
MANIFEST_NAME = "vllm-hust-extension-v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single(paths: list[Path], kind: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {kind}, found {len(paths)}")
    return paths[0]


def _regular_file(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return not mode or stat.S_ISREG(mode)


def validate_distribution(dist: Path) -> dict[str, object]:
    wheel = _single(sorted(dist.glob("*.whl")), "wheel")
    sdist = _single(sorted(dist.glob("*.tar.gz")), "sdist")

    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        manifest_entries = [
            item for item in entries if Path(item.filename).name == MANIFEST_NAME
        ]
        manifest = _single(manifest_entries, "Bundle manifest")
        if not _regular_file(manifest):
            raise ValueError("Bundle manifest must be a regular file")

        payload = json.loads(archive.read(manifest))
        if payload.get("bundle_id") != EXPECTED_BUNDLE_ID:
            raise ValueError("Bundle manifest has an unexpected bundle_id")
        if payload.get("bundle_version") != EXPECTED_VERSION:
            raise ValueError("Bundle manifest has an unexpected bundle_version")

        metadata_entry = _single(
            [item for item in entries if item.filename.endswith(".dist-info/METADATA")],
            "wheel METADATA",
        )
        metadata = BytesParser().parsebytes(archive.read(metadata_entry))
        if metadata.get("Name") != EXPECTED_DISTRIBUTION:
            raise ValueError("wheel METADATA has an unexpected Name")
        if metadata.get("Version") != EXPECTED_VERSION:
            raise ValueError("wheel METADATA has an unexpected Version")

        entry_points_entry = _single(
            [
                item
                for item in entries
                if item.filename.endswith(".dist-info/entry_points.txt")
            ],
            "entry_points.txt",
        )
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read_string(archive.read(entry_points_entry).decode("utf-8"))
        if set(parser.sections()) != {"vllm.extension_bundles"}:
            raise ValueError("wheel must expose only the static Bundle entry point")
        if dict(parser["vllm.extension_bundles"]) != {
            EXPECTED_BUNDLE_ID: EXPECTED_ENTRY_POINT
        }:
            raise ValueError("wheel has an unexpected Bundle entry point")

    return {
        "schema_version": "adaptive-quantized-kv-distribution-validation/v1",
        "status": "PASS",
        "distribution": EXPECTED_DISTRIBUTION,
        "version": EXPECTED_VERSION,
        "bundle_id": EXPECTED_BUNDLE_ID,
        "wheel": {"name": wheel.name, "sha256": _sha256(wheel)},
        "sdist": {"name": sdist.name, "sha256": _sha256(sdist)},
        "runtime_activation_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        receipt = validate_distribution(args.dist)
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
