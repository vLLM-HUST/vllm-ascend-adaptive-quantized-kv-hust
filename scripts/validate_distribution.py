#!/usr/bin/env python3
"""Validate an inert extension distribution without importing vLLM."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import posixpath
import re
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

EXPECTED_DISTRIBUTION = "vllm-ascend-adaptive-quantized-kv-hust"
EXPECTED_VERSION = "0.1.0.dev0"
EXPECTED_EXTENSION_ID = "org.vllm-hust.ascend-adaptive-quantized-kv"
EXPECTED_ENTRY_POINT = "vllm_ascend_adaptive_quantized_kv"
ENTRY_POINT_GROUP = "vllm_hust.extension_bundles"
MANIFEST_NAME = "vllm-hust-extension-v0.3.json"


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


def validate_sdist_links(sdist: Path) -> dict[str, int]:
    """Check packaged Markdown destinations without extracting archive members."""
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        roots = {item.name.partition("/")[0] for item in members}
        if len(roots) != 1:
            raise ValueError("sdist must have one root")
        files = {}
        for item in members:
            relative = item.name.partition("/")[2]
            if (
                item.name.startswith("/")
                or ".." in item.name.split("/")
                or item.issym()
                or item.islnk()
            ):
                raise ValueError("sdist contains unsafe or linked members")
            if item.isfile():
                if not relative or relative in files:
                    raise ValueError("sdist contains missing or duplicate file names")
                files[relative] = item
        links = 0
        for name, item in files.items():
            if not name.endswith(".md"):
                continue
            if item.size > 2 * 1024 * 1024:
                raise ValueError("sdist Markdown exceeds byte cap")
            stream = archive.extractfile(item)
            if stream is None:
                raise ValueError("sdist Markdown is not readable")
            with stream:
                markdown = stream.read(2 * 1024 * 1024 + 1).decode("utf-8")
            for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", markdown):
                parsed = urlsplit(target.strip("<>"))
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                path = unquote(parsed.path)
                destination = posixpath.normpath(
                    posixpath.join(posixpath.dirname(name), path)
                )
                if path.startswith("/") or destination not in files:
                    raise ValueError(f"sdist local link missing: {name} -> {target}")
                links += 1
        return {"files": len(files), "local_file_links": links}


def validate_distribution(dist: Path) -> dict[str, object]:
    wheel = _single(sorted(dist.glob("*.whl")), "wheel")
    sdist = _single(sorted(dist.glob("*.tar.gz")), "sdist")
    sdist_links = validate_sdist_links(sdist)

    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        manifest_entries = [
            item for item in entries if Path(item.filename).name == MANIFEST_NAME
        ]
        manifest = _single(manifest_entries, "extension manifest")
        if not _regular_file(manifest):
            raise ValueError("extension manifest must be a regular file")

        payload = json.loads(archive.read(manifest))
        if payload.get("schema_version") != "0.3-experimental":
            raise ValueError("extension manifest has an unexpected schema_version")
        if payload.get("extension_id") != EXPECTED_EXTENSION_ID:
            raise ValueError("extension manifest has an unexpected extension_id")
        if payload.get("extension_version") != EXPECTED_VERSION:
            raise ValueError("extension manifest has an unexpected extension_version")
        if payload.get("implementation", [{}])[0].get("status") != "import_only":
            raise ValueError("P1 extension implementation must remain import_only")

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
        if set(parser.sections()) != {ENTRY_POINT_GROUP}:
            raise ValueError("wheel must expose only the static extension entry point")
        if dict(parser[ENTRY_POINT_GROUP]) != {
            EXPECTED_EXTENSION_ID: EXPECTED_ENTRY_POINT
        }:
            raise ValueError("wheel has an unexpected extension entry point")

    return {
        "schema_version": "adaptive-quantized-kv-distribution-validation/v1",
        "status": "PASS",
        "distribution": EXPECTED_DISTRIBUTION,
        "version": EXPECTED_VERSION,
        "extension_id": EXPECTED_EXTENSION_ID,
        "wheel": {"name": wheel.name, "sha256": _sha256(wheel)},
        "sdist": {
            "name": sdist.name,
            "sha256": _sha256(sdist),
            "packaged_links": sdist_links,
        },
        "runtime_activation_present": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        receipt = validate_distribution(args.dist)
    except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
        parser.error(str(error))

    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
