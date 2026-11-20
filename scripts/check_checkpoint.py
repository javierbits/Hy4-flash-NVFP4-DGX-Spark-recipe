#!/usr/bin/env python3
"""Checks a local Hy4-flash NVFP4 checkpoint and resolves its directory.

Reads ``config.json`` and ``model.safetensors.index.json`` only. Does not
map safetensors, does not import torch, and does not start a server.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class Expectation:
    """Counts a finished expert-NVFP4 Hy4-flash directory must match.

    packed: ``*.weight_packed`` tensors, one per routed-expert shard.
    scale: ``*.weight_scale`` tensors, same count as packed.
    global_scale: ``*.weight_global_scale`` tensors, same count as packed.
    engram_tables: BF16 Engram lookup tables left unpacked.
    total_size: ``metadata.total_size`` in the safetensors index, in bytes.
    """

    packed: int
    scale: int
    global_scale: int
    engram_tables: int
    total_size: int


# Finished checkpoint: index metadata total_size 226938653558,
# 82 expert shards, 4 Engram lookup tables left in BF16.
PRODUCTION = Expectation(
    packed=82,
    scale=82,
    global_scale=82,
    engram_tables=4,
    total_size=226_938_653_558,
)

# Hugging Face repo id. The snapshot lives in the hub cache, not a fixed path.
REPO = "javierbits/Hy4-flash-NVFP4"

INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"


@dataclass
class Report:
    """Outcome of checking one checkpoint directory.

    ok: True when every required field matches the expectation.
    errors: Human-readable failures. Empty when ok is True.
    path: Directory that was checked.
    total_size: Index metadata total_size, or None when the index was unread.
    packed: Count of .weight_packed names.
    scale: Count of .weight_scale names.
    global_scale: Count of .weight_global_scale names.
    engram_tables: Count of Engram lookup_table names.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    path: Path | None = None
    total_size: int | None = None
    packed: int = 0
    scale: int = 0
    global_scale: int = 0
    engram_tables: int = 0


def hub_root() -> Path:
    """Returns the Hugging Face hub cache directory.

    ``HF_HUB_CACHE`` wins, then ``$HF_HOME/hub``, then
    ``~/.cache/huggingface/hub``.
    """

    override = os.environ.get("HF_HUB_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return Path(home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_checkpoint(path: Path) -> bool:
    """Returns True when ``path`` has both config.json and the index."""

    return (path / CONFIG_NAME).is_file() and (path / INDEX_NAME).is_file()


def snapshot_of(repo: str) -> Path | None:
    """Returns the snapshot directory for ``repo`` in the hub cache.

    Prefers ``refs/main``. Otherwise the only snapshot that holds both
    checkpoint files. Returns None when the repo is not cached.
    """

    root = hub_root() / f"models--{repo.replace('/', '--')}"
    snaps = root / "snapshots"
    ref = root / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text(encoding="utf-8").strip()
        pinned = snaps / revision
        if is_checkpoint(pinned):
            return pinned
    if not snaps.is_dir():
        return None
    found = [child for child in sorted(snaps.iterdir()) if is_checkpoint(child)]
    if len(found) == 1:
        return found[0]
    return None


def find_checkpoint() -> Path | None:
    """Returns the checkpoint directory for this recipe.

    ``MODEL_PATH`` may be a directory lmdeck already resolved, or a
    ``org/name`` repo id. Otherwise the recipe repo is resolved from the
    hub cache.
    """

    raw = os.environ.get("MODEL_PATH", "").strip()
    if raw:
        path = Path(raw).expanduser()
        if is_checkpoint(path):
            return path
        if raw.count("/") == 1 and not raw.startswith(("/", "~")):
            found = snapshot_of(raw)
            if found is not None:
                return found
    return snapshot_of(REPO)


def _load_json(path: Path) -> object:
    """Reads one JSON file. Raises OSError or json.JSONDecodeError."""

    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _quant_block(config: dict) -> dict | None:
    """Returns quantization_config when it is an object."""

    block = config.get("quantization_config")
    if isinstance(block, dict):
        return block
    return None


def _weight_names(weight_map: dict) -> list[str]:
    """Returns tensor names from an index weight_map."""

    return [name for name in weight_map if isinstance(name, str)]


def check(path: Path, expect: Expectation = PRODUCTION) -> Report:
    """Checks ``path`` against ``expect``.

    A missing file, a bad quant block, a bare expert parameter name, or a
    count mismatch becomes an error. The function never raises for a
    malformed checkpoint; those cases are reported.
    """

    report = Report(ok=False, path=path)
    config_path = path / CONFIG_NAME
    index_path = path / INDEX_NAME
    if not config_path.is_file() or not index_path.is_file():
        report.errors.append(f"{path} is missing {CONFIG_NAME} or {INDEX_NAME}")
        return report
    try:
        config = _load_json(config_path)
        index = _load_json(index_path)
    except (OSError, json.JSONDecodeError) as exc:
        report.errors.append(f"unreadable checkpoint json: {exc}")
        return report
    if not isinstance(config, dict) or not isinstance(index, dict):
        report.errors.append("config.json and the index must be JSON objects")
        return report

    root = _quant_block(config)
    text = config.get("text_config")
    text_block = _quant_block(text) if isinstance(text, dict) else None
    if root is None:
        report.errors.append("config.json has no quantization_config object")
    else:
        _check_quant_block(root, report)
    if text_block is None:
        report.errors.append("text_config.quantization_config is missing")
    elif root is not None and text_block != root:
        report.errors.append("text_config.quantization_config differs from the root block")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        report.errors.append("index weight_map is missing")
        weight_map = {}
    names = _weight_names(weight_map)
    report.packed = sum(1 for name in names if name.endswith(".weight_packed"))
    report.scale = sum(1 for name in names if name.endswith(".weight_scale"))
    report.global_scale = sum(1 for name in names if name.endswith(".weight_global_scale"))
    report.engram_tables = sum(1 for name in names if name.endswith("engram_layer.lookup_table"))
    bare = [
        name
        for name in names
        if name.endswith("experts.gate_up_proj") or name.endswith("experts.down_proj")
    ]
    if bare:
        report.errors.append(f"bare expert parameters are still in the index ({bare[0]})")
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("total_size"), int):
        report.total_size = metadata["total_size"]
    else:
        report.errors.append("index metadata.total_size is missing")
    _expect_count(report, "weight_packed", report.packed, expect.packed)
    _expect_count(report, "weight_scale", report.scale, expect.scale)
    _expect_count(report, "weight_global_scale", report.global_scale, expect.global_scale)
    _expect_count(report, "engram lookup tables", report.engram_tables, expect.engram_tables)
    if report.total_size is not None and report.total_size != expect.total_size:
        report.errors.append(
            f"total_size is {report.total_size}, expected {expect.total_size}"
        )
    report.ok = not report.errors
    return report


def _expect_count(report: Report, label: str, got: int, want: int) -> None:
    """Appends an error when ``got`` is not ``want``."""

    if got != want:
        report.errors.append(f"{label} count is {got}, expected {want}")


def _check_quant_block(block: dict, report: Report) -> None:
    """Appends errors when the compressed-tensors NVFP4 block is wrong.

    Requires W4A16 (``input_activations`` null), group size 16, and the
    fused expert target regex. Engram and shared experts must stay in
    the ignore list.
    """

    if block.get("quant_method") != "compressed-tensors":
        report.errors.append("quant_method is not compressed-tensors")
    if block.get("format") != "nvfp4-pack-quantized":
        report.errors.append("format is not nvfp4-pack-quantized")
    if block.get("quantization_status") != "compressed":
        report.errors.append("quantization_status is not compressed")
    groups = block.get("config_groups")
    group = groups.get("NVFP4A16") if isinstance(groups, dict) else None
    if not isinstance(group, dict):
        report.errors.append("config_groups.NVFP4A16 is missing")
        return
    if group.get("input_activations") is not None:
        report.errors.append("input_activations is set; this checkpoint is W4A16")
    weights = group.get("weights")
    if not isinstance(weights, dict):
        report.errors.append("NVFP4A16.weights is missing")
    else:
        if weights.get("num_bits") != 4:
            report.errors.append("NVFP4A16 weight num_bits is not 4")
        if weights.get("group_size") != 16:
            report.errors.append("NVFP4A16 weight group_size is not 16")
    targets = group.get("targets")
    target_text = " ".join(targets) if isinstance(targets, list) else ""
    if "gate_up_proj" not in target_text or "down_proj" not in target_text:
        report.errors.append("NVFP4A16 targets do not name gate_up_proj and down_proj")
    ignored = block.get("ignore")
    ignore_text = " ".join(ignored) if isinstance(ignored, list) else ""
    if "engram" not in ignore_text or "shared_experts" not in ignore_text:
        report.errors.append("ignore list drops Engram or shared experts")


def format_report(report: Report) -> str:
    """Formats a report as a short multi-line status."""

    where = str(report.path) if report.path is not None else "(no path)"
    lines = [
        f"path: {where}",
        f"ok: {str(report.ok).lower()}",
        f"packed: {report.packed}",
        f"scale: {report.scale}",
        f"global_scale: {report.global_scale}",
        f"engram_tables: {report.engram_tables}",
        f"total_size: {report.total_size}",
    ]
    for error in report.errors:
        lines.append(f"error: {error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry. ``--find`` prints the resolved directory.

    Returns 0 when the check passes or a directory is found. Returns 1
    when the checkpoint is missing or does not match. Returns 2 on a
    bad invocation.
    """

    parser = argparse.ArgumentParser(description="Check a Hy4-flash NVFP4 checkpoint directory.")
    parser.add_argument("path", nargs="?", help="Checkpoint directory. Default: the hub cache snapshot.")
    parser.add_argument(
        "--find",
        action="store_true",
        help="Print the first directory that contains config.json and the index.",
    )
    args = parser.parse_args(argv)
    if args.find and args.path:
        print("pass either --find or a path", file=sys.stderr)
        return 2
    if args.find:
        found = find_checkpoint()
        if found is None:
            print(f"{REPO} is not in {hub_root()}", file=sys.stderr)
            return 1
        print(found)
        return 0
    path = Path(args.path).expanduser() if args.path else find_checkpoint()
    if path is None:
        print(f"{REPO} is not in {hub_root()}", file=sys.stderr)
        return 1
    report = check(path)
    stream = sys.stdout if report.ok else sys.stderr
    print(format_report(report), file=stream)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
