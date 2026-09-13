"""Download CUAD v1 and build deterministic corpus manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

CUAD_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
CUAD_MD5 = "c38f490a984420b8a62600db401fafd5"

_CONTRACT_TYPES = {
    "affiliate agreement": "Affiliate Agreement",
    "affiliate agreements": "Affiliate Agreement",
    "agency agreements": "Agency Agreement",
    "co branding": "Co-Branding Agreement",
    "collaboration": "Collaboration/Cooperation Agreement",
    "consulting agreements": "Consulting Agreement",
    "development": "Development Agreement",
    "distributor": "Distributor Agreement",
    "endorsement": "Endorsement Agreement",
    "endorsement agreement": "Endorsement Agreement",
    "franchise": "Franchise Agreement",
    "hosting": "Hosting Agreement",
    "ip": "IP Agreement",
    "joint venture": "Joint Venture Agreement",
    "joint venture filing": "Joint Venture Agreement",
    "license agreements": "License Agreement",
    "maintenance": "Maintenance Agreement",
    "manufacturing": "Manufacturing Agreement",
    "marketing": "Marketing Agreement",
    "non compete non solicit": "Non-Compete/No-Solicit/Non-Disparagement Agreement",
    "outsourcing": "Outsourcing Agreement",
    "promotion": "Promotion Agreement",
    "reseller": "Reseller Agreement",
    "service": "Service Agreement",
    "sponsorship": "Sponsorship Agreement",
    "strategic alliance": "Strategic Alliance Agreement",
    "supply": "Supply Agreement",
    "transportation": "Transportation Agreement",
}


@dataclass(frozen=True)
class CuadRecord:
    """One CUAD contract and its paired source files."""

    doc_id: str
    title: str
    contract_type: str
    part: str
    pdf_path: str
    txt_path: str
    pdf_sha256: str
    pdf_size_bytes: int


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the hexadecimal MD5 checksum used by the Zenodo record."""
    digest = hashlib.md5()  # noqa: S324 - integrity check against the publisher's checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path, expected_md5: str) -> Path:
    """Download to a partial file, resume when possible, and atomically finalize."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and file_md5(destination) == expected_md5:
        return destination

    partial = destination.with_suffix(f"{destination.suffix}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "contract-rag/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - configured trusted URL
        append = offset > 0 and response.status == 206
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)

    actual_md5 = file_md5(partial)
    if actual_md5 != expected_md5:
        partial.unlink()
        raise ValueError(f"CUAD checksum mismatch: expected {expected_md5}, got {actual_md5}")
    os.replace(partial, destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> Path:
    """Safely and idempotently extract CUAD, skipping complete files."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in CUAD archive: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    return destination / "CUAD_v1"


def _match_key(filename: str) -> str:
    stem = unicodedata.normalize("NFKD", Path(filename).stem).casefold()
    return re.sub(r"[^a-z0-9]", "", stem)


def _slug(filename: str) -> str:
    stem = unicodedata.normalize("NFKD", Path(filename).stem)
    ascii_stem = stem.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", ascii_stem).strip("-")


def _canonical_contract_type(folder_name: str) -> str:
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", folder_name.casefold())).strip()
    try:
        return _CONTRACT_TYPES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown CUAD contract-type folder: {folder_name}") from exc


def _index_files(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[_match_key(path.name)].append(path)
    return index


def _unique_match(filename: str, index: dict[str, list[Path]], kind: str) -> Path:
    key = _match_key(filename)
    candidates = index.get(key, [])
    if not candidates:
        candidates = [
            path
            for other, paths in index.items()
            if key.startswith(other) or other.startswith(key)
            for path in paths
        ]
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates[:5]) or "none"
        raise ValueError(f"Expected one {kind} match for {filename!r}; found {len(candidates)}: {rendered}")
    return candidates[0]


def build_manifest(cuad_root: Path, repo_root: Path) -> list[CuadRecord]:
    """Reconcile the CSV, PDF, and TXT inventories into one stable manifest."""
    pdf_root = cuad_root / "full_contract_pdf"
    txt_root = cuad_root / "full_contract_txt"
    pdf_index = _index_files([p for p in pdf_root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"])
    txt_index = _index_files([p for p in txt_root.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"])

    with (cuad_root / "master_clauses.csv").open(encoding="utf-8-sig", newline="") as handle:
        filenames = [row["Filename"].strip() for row in csv.DictReader(handle)]

    records: list[CuadRecord] = []
    for filename in filenames:
        pdf = _unique_match(filename, pdf_index, "PDF")
        txt = _unique_match(str(Path(filename).with_suffix(".txt")), txt_index, "TXT")
        relative = pdf.relative_to(pdf_root)
        records.append(
            CuadRecord(
                doc_id=_slug(filename),
                title=Path(filename).stem,
                contract_type=_canonical_contract_type(relative.parts[1]),
                part=relative.parts[0],
                pdf_path=pdf.relative_to(repo_root).as_posix(),
                txt_path=txt.relative_to(repo_root).as_posix(),
                pdf_sha256=file_sha256(pdf),
                pdf_size_bytes=pdf.stat().st_size,
            )
        )

    records.sort(key=lambda record: record.doc_id)
    if len(records) != 510 or len({record.doc_id for record in records}) != 510:
        raise ValueError("CUAD manifest must contain 510 unique document IDs")
    return records


def stratified_subset(records: list[CuadRecord], size: int, seed: int) -> list[CuadRecord]:
    """Select a deterministic proportional sample, including each stratum when possible."""
    if not 0 < size <= len(records):
        raise ValueError("Subset size must be between 1 and the manifest size")

    groups: dict[str, list[CuadRecord]] = defaultdict(list)
    for record in records:
        groups[record.contract_type].append(record)
    require_each = size >= len(groups)
    exact = {name: len(group) * size / len(records) for name, group in groups.items()}
    allocation = {name: max(int(exact[name]), int(require_each)) for name in groups}

    while sum(allocation.values()) < size:
        choices = sorted(groups, key=lambda name: (-(exact[name] - allocation[name]), name))
        allocation[choices[0]] += 1
    while sum(allocation.values()) > size:
        choices = sorted(
            (name for name in groups if allocation[name] > int(require_each)),
            key=lambda name: (exact[name] - allocation[name], name),
        )
        allocation[choices[0]] -= 1

    rng = random.Random(seed)
    selected: list[CuadRecord] = []
    for name in sorted(groups):
        candidates = sorted(groups[name], key=lambda record: record.doc_id)
        rng.shuffle(candidates)
        selected.extend(candidates[: allocation[name]])
    return sorted(selected, key=lambda record: record.doc_id)


def write_jsonl(records: list[CuadRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def build_fixture(
    records: list[CuadRecord], repo_root: Path, destination: Path, size: int = 3
) -> list[CuadRecord]:
    """Copy the smallest documents from distinct types into a compact test corpus."""
    chosen: list[CuadRecord] = []
    seen_types: set[str] = set()
    for record in sorted(records, key=lambda item: (item.pdf_size_bytes, item.doc_id)):
        if record.contract_type in seen_types:
            continue
        chosen.append(record)
        seen_types.add(record.contract_type)
        if len(chosen) == size:
            break

    destination.mkdir(parents=True, exist_ok=True)
    fixture_records: list[CuadRecord] = []
    for record in chosen:
        pdf_source = repo_root / record.pdf_path
        txt_source = repo_root / record.txt_path
        pdf_target = destination / "pdf" / pdf_source.name
        txt_target = destination / "txt" / txt_source.name
        pdf_target.parent.mkdir(parents=True, exist_ok=True)
        txt_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf_source, pdf_target)
        shutil.copyfile(txt_source, txt_target)
        fixture_records.append(
            CuadRecord(
                **{
                    **asdict(record),
                    "pdf_path": pdf_target.relative_to(repo_root).as_posix(),
                    "txt_path": txt_target.relative_to(repo_root).as_posix(),
                }
            )
        )
    write_jsonl(fixture_records, destination / "manifest.jsonl")
    return fixture_records


def run(config: dict[str, Any], repo_root: Path, *, create_fixture: bool = False) -> tuple[int, int]:
    raw_dir = repo_root / config["raw_dir"]
    archive = download_archive(config["url"], raw_dir / "CUAD_v1.zip", config["md5"])
    cuad_root = extract_archive(archive, raw_dir)
    records = build_manifest(cuad_root, repo_root)
    subset = stratified_subset(records, int(config["dev_subset_size"]), int(config["seed"]))
    write_jsonl(records, repo_root / config["manifest_path"])
    write_jsonl(subset, repo_root / config["dev_manifest_path"])
    if create_fixture:
        build_fixture(records, repo_root, repo_root / config["fixture_dir"])
    return len(records), len(subset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument(
        "--create-fixture", action="store_true", help="Refresh the committed three-document fixture"
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    total, dev = run(config, repo_root, create_fixture=args.create_fixture)
    print(f"Wrote manifest for {total} CUAD contracts and deterministic dev subset of {dev}.")


if __name__ == "__main__":
    main()
