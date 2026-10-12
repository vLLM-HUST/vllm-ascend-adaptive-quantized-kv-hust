import io
import tarfile

import pytest

from scripts.validate_distribution import validate_sdist_links


def make_archive(tmp_path, files):
    path = tmp_path / "package.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files:
            member = tarfile.TarInfo(name)
            raw = data.encode()
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return path


def test_packaged_links_use_archive_not_worktree(tmp_path):
    path = make_archive(
        tmp_path,
        [
            ("package/README.md", "[meta](MOD_METADATA.json) [doc](docs/cost.md)"),
            ("package/MOD_METADATA.json", "{}"),
            (
                "package/docs/cost.md",
                "[readme](../README.md#scope) [web](https://example.org)",
            ),
        ],
    )
    assert validate_sdist_links(path) == {"files": 3, "local_file_links": 3}


@pytest.mark.parametrize(
    "target", ["MOD_METADATA.json", "../outside.md", "/absolute.md", "%2e%2e/out.md"]
)
def test_missing_or_escaping_packaged_link_rejected(tmp_path, target):
    (tmp_path / "MOD_METADATA.json").write_text("{}")
    path = make_archive(tmp_path, [("package/README.md", f"[target]({target})")])
    with pytest.raises(ValueError, match="local link missing"):
        validate_sdist_links(path)


@pytest.mark.parametrize(
    "files",
    [
        [("one/a.md", ""), ("two/b.md", "")],
        [("package/../a.md", "")],
        [("/package/a.md", "")],
        [("package/a.md", ""), ("package/a.md", "")],
    ],
)
def test_invalid_archive_member_identity_rejected(tmp_path, files):
    path = make_archive(tmp_path, files)
    with pytest.raises(ValueError):
        validate_sdist_links(path)


def test_linked_archive_member_rejected_without_extracting(tmp_path):
    path = tmp_path / "linked.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("package/README.md")
        member.type = tarfile.SYMTYPE
        member.linkname = "/outside"
        archive.addfile(member)
    with pytest.raises(ValueError, match="unsafe or linked"):
        validate_sdist_links(path)
