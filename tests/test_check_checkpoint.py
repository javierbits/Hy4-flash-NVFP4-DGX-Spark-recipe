"""Tests for the Hy4-flash NVFP4 checkpoint checker and the serve guard."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.check_checkpoint import (
    PRODUCTION,
    REPO,
    Expectation,
    check,
    find_checkpoint,
    main,
)

SMALL = Expectation(packed=1, scale=1, global_scale=1, engram_tables=1, total_size=64)


def _quant_block() -> dict:
    """Returns a minimal NVFP4A16 block that check() accepts."""

    return {
        "config_groups": {
            "NVFP4A16": {
                "format": "nvfp4-pack-quantized",
                "input_activations": None,
                "targets": [r"re:model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)$"],
                "weights": {"group_size": 16, "num_bits": 4},
            }
        },
        "format": "nvfp4-pack-quantized",
        "ignore": ["re:.*shared_experts.*", "re:.*engram.*"],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def _write_checkpoint(directory: Path, *, bare: str | None = None, drop_text: bool = False) -> None:
    """Writes a one-expert fixture checkpoint into ``directory``."""

    block = _quant_block()
    config: dict = {"quantization_config": block}
    if not drop_text:
        config["text_config"] = {"quantization_config": json.loads(json.dumps(block))}
    weight_map = {
        "model.layers.1.mlp.experts.gate_up_proj.weight_packed": "model-00001.safetensors",
        "model.layers.1.mlp.experts.gate_up_proj.weight_scale": "model-00001.safetensors",
        "model.layers.1.mlp.experts.gate_up_proj.weight_global_scale": "model-00001.safetensors",
        "model.layers.1.engram_layer.lookup_table": "model-00002.safetensors",
    }
    if bare is not None:
        weight_map[bare] = "model-00003.safetensors"
    index = {"metadata": {"total_size": SMALL.total_size}, "weight_map": weight_map}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")


class CheckCheckpointTest(unittest.TestCase):
    """Covers a good fixture, a bare expert name, and a missing directory."""

    def test_small_fixture_passes(self) -> None:
        """A one-shard fixture matches an Expectation of the same counts."""

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory)
            report = check(directory, SMALL)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.packed, 1)
        self.assertEqual(report.total_size, 64)

    def test_bare_expert_name_fails(self) -> None:
        """A leftover BF16 expert parameter name is a failure."""

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, bare="model.layers.2.mlp.experts.down_proj")
            report = check(directory, SMALL)
        self.assertFalse(report.ok)
        self.assertTrue(any("bare expert" in error for error in report.errors))

    def test_missing_directory_fails(self) -> None:
        """A path with no config or index fails without raising."""

        report = check(Path(tempfile.mkdtemp()) / "absent", SMALL)
        self.assertFalse(report.ok)
        self.assertTrue(report.errors)

    def test_text_config_must_carry_the_same_block(self) -> None:
        """The root quant block has to be copied onto text_config."""

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_checkpoint(directory, drop_text=True)
            report = check(directory, SMALL)
        self.assertFalse(report.ok)
        self.assertTrue(any("text_config" in error for error in report.errors))

    def test_find_reads_the_hub_snapshot(self) -> None:
        """A repo id in MODEL_PATH resolves through the hub cache."""

        with tempfile.TemporaryDirectory() as tmp:
            revision = "a" * 40
            snap = Path(tmp) / "models--javierbits--Hy4-flash-NVFP4" / "snapshots" / revision
            snap.mkdir(parents=True)
            _write_checkpoint(snap)
            refs = Path(tmp) / "models--javierbits--Hy4-flash-NVFP4" / "refs"
            refs.mkdir()
            (refs / "main").write_text(revision + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["HF_HUB_CACHE"] = tmp
            env["MODEL_PATH"] = REPO
            env.pop("HF_HOME", None)
            found = find_checkpoint_in_env(env)
        self.assertEqual(found, snap)

    def test_find_skips_a_model_path_that_has_no_index(self) -> None:
        """MODEL_PATH that is an empty directory falls through to the hub snapshot."""

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            revision = "b" * 40
            snap = Path(tmp) / "hub" / "models--javierbits--Hy4-flash-NVFP4" / "snapshots" / revision
            snap.mkdir(parents=True)
            _write_checkpoint(snap)
            env = os.environ.copy()
            env["HF_HUB_CACHE"] = str(Path(tmp) / "hub")
            env["MODEL_PATH"] = str(empty)
            env.pop("HF_HOME", None)
            found = find_checkpoint_in_env(env)
        self.assertEqual(found, snap)

    def test_production_expectation_matches_the_known_checkpoint_math(self) -> None:
        """The shipped expectation is the finished index, not a rounded guess."""

        self.assertEqual(PRODUCTION.total_size, 226_938_653_558)
        self.assertEqual(PRODUCTION.packed, 82)
        self.assertEqual(PRODUCTION.engram_tables, 4)

    def test_cached_checkpoint_matches_production(self) -> None:
        """The hub snapshot matches the production expectation when it is cached."""

        env = os.environ.copy()
        env.pop("HF_HUB_CACHE", None)
        env.pop("HF_HOME", None)
        env.pop("MODEL_PATH", None)
        found = find_checkpoint_in_env(env)
        if found is None:
            self.skipTest(f"{REPO} is not in the Hugging Face cache")
        report = check(found)
        self.assertTrue(report.ok, report.errors)


class ServeCommandTest(unittest.TestCase):
    """The recipe launches vLLM on four nodes."""

    def test_toml_starts_vllm_on_four_nodes(self) -> None:
        """lmdeck.toml runs vllm serve once per rank and names the hub repo."""

        text = (ROOT / "lmdeck.toml").read_text(encoding="utf-8")
        self.assertEqual(text.count("exec vllm serve javierbits/Hy4-flash-NVFP4"), 4)
        self.assertEqual(text.count('"docker", "run"'), 4)
        for rank in range(4):
            self.assertIn(f"--node-rank {rank}", text)
        self.assertIn('host = "local"', text)
        self.assertNotIn("exit 2", text)
        self.assertNotIn("/home/", text)
        # This vLLM build rejects DFly unless the checkpoint is FP8 W8A8.
        # The NVFP4 W4A16 weights fail that check before the server binds.
        self.assertNotIn("speculative-config", text)
        self.assertNotIn("dfly", text)

    def test_find_cli_prints_the_hub_snapshot(self) -> None:
        """--find prints the snapshot directory from HF_HUB_CACHE."""

        with tempfile.TemporaryDirectory() as tmp:
            revision = "c" * 40
            snap = Path(tmp) / "models--javierbits--Hy4-flash-NVFP4" / "snapshots" / revision
            snap.mkdir(parents=True)
            _write_checkpoint(snap)
            refs = snap.parents[1] / "refs"
            refs.mkdir()
            (refs / "main").write_text(revision + "\n", encoding="utf-8")
            env = os.environ.copy()
            env["HF_HUB_CACHE"] = tmp
            env["MODEL_PATH"] = ""
            env.pop("HF_HOME", None)
            result = subprocess.run(
                ["python3", str(ROOT / "scripts" / "check_checkpoint.py"), "--find"],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                cwd=ROOT,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(snap))


def find_checkpoint_in_env(env: dict[str, str]) -> Path | None:
    """Runs find_checkpoint with ``env`` patched into os.environ."""

    saved = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        return find_checkpoint()
    finally:
        os.environ.clear()
        os.environ.update(saved)


class MainCliTest(unittest.TestCase):
    """main() returns 1 for a missing path and 2 for a mixed invocation."""

    def test_main_rejects_find_plus_path(self) -> None:
        """--find together with a path is a usage error."""

        self.assertEqual(main(["--find", "/tmp"]), 2)

    def test_main_missing_path_returns_1(self) -> None:
        """A directory without the index fails closed."""

        with tempfile.TemporaryDirectory() as tmp:
            code = main([tmp])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
