# Hy4-flash NVFP4 W4A16

lmdeck recipe for [`javierbits/Hy4-flash-NVFP4`](https://huggingface.co/javierbits/Hy4-flash-NVFP4), an NVFP4 checkpoint of the routed experts in [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash). This repo is the recipe. The weights live in the Hugging Face cache, not in this git repo.

```sh
lmdeck add https://github.com/javierbits/Hy4-flash-NVFP4-DGX-Spark-recipe
lmdeck use hy4-flash-nvfp4-dgx-spark-recipe
```

`lmdeck add` names the deck entry from the repo and injects `[source]`. The manifest leaves `[source]` out on purpose. `lmdeck use` starts `vllm/vllm-openai:hy4-flash` across four machines, tensor parallel 4, and serves `hy4-flash` on port 8891. DFly speculative decoding is left off: this vLLM image accepts it only for FP8 W8A8 checkpoints, and these weights are NVFP4 W4A16. The weights are `javierbits/Hy4-flash-NVFP4` in the Hugging Face hub cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`). lmdeck mounts that cache from the machine that already has the snapshot.

## Quant

Data-free round-to-nearest NVFP4A16 in the compressed-tensors layout (`memoryless_minmax`, group size 16, FP8 E4M3 local scales, one FP32 global scale per expert). Activations stay BF16. The quantized tensors are the fused routed experts `model.layers.*.mlp.experts.gate_up_proj` and `down_proj`. Attention, layer 0, shared experts, `routed_expert_*`, Engram, vision, and DFly2 stay BF16.

The index `metadata.total_size` is 226938653558 (211.35 GiB): 82 packed expert shards and 4 Engram lookup tables still in BF16. One 128 GB machine cannot hold that with the Engram tables resident. The recipe asks for four machines and 96 GB on each.

Packed names:

```text
model.layers.N.mlp.experts.gate_up_proj.weight_packed
model.layers.N.mlp.experts.gate_up_proj.weight_scale
model.layers.N.mlp.experts.gate_up_proj.weight_global_scale
model.layers.N.mlp.experts.down_proj.weight_packed
model.layers.N.mlp.experts.down_proj.weight_scale
model.layers.N.mlp.experts.down_proj.weight_global_scale
```

`weight_packed` is uint8, two E2M1 codes per byte, low nibble first. `weight_scale` is float8_e4m3fn. `weight_global_scale` is float32 with shape `[experts, 1]`. Reconstruct with `code * scale / global_scale`.

The weights stay under the Tencent Hy license that ships inside the checkpoint. This recipe repo does not redistribute them.

## Check

```sh
python3 -m unittest tests/test_check_checkpoint.py -v
python3 scripts/check_checkpoint.py --find
```

The checkpoint test runs when `javierbits/Hy4-flash-NVFP4` is in the local Hugging Face cache. Other machines skip it. `lmdeck use --dry-run` prints the four-node plan without starting vLLM.
