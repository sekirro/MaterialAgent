# MaterialAgent

MaterialAgent is a part-level material-parameter agent for PhysGM workflows. It consumes outputs from
[`sekirro/PartPhysAgent`](https://github.com/sekirro/PartPhysAgent), estimates each part's material,
Young's modulus `E`, Poisson's ratio `nu`, and density, samples candidate parameter sets, runs PhysGM
candidate simulations, scores rendered videos, and writes the selected simulation config.

## What It Does

- Loads PartPhysAgent scene outputs, including part schema, masks, AABBs, Gaussian part ids, and whole-object PhysGM metadata.
- Extracts per-part PhysGM distributions from part crops, including `E_mu/E_var` and `nu_mu/nu_var`.
- Optionally queries an OpenAI-compatible VLM for visual material priors per part.
- Builds a posterior over material, `E`, and `nu` for every part.
- Samples candidate parameter sets such as MAP, soft-response, stiff-response, uncertain sweeps, and material alternatives.
- Compiles simulation configs for either:
  - `part_id`: per-Gaussian part material maps for PartPhysAgent's `gs_simulation_partid_materials.py`
  - `aabb`: PhysGM-compatible `additional_material_params`
- Runs candidate simulations, checks rendered video outputs, and selects the best viable candidate.

## Repository Layout

```text
material_agent/
  cli.py                         # command-line entry point
  loaders/partphys_outputs.py    # PartPhysAgent output loader
  evidence/physgm_distribution.py
  evidence/vlm_material.py
  reasoning/posterior.py
  reasoning/candidate_sampler.py
  simulation/config_compiler.py
  simulation/runner.py
  evaluation/video_metrics.py
configs/
  default.yaml
  stable_solver.yaml
tests/
  test_material_agent_core.py
```

## Quick Start On The Server

```bash
conda activate physgm
PYTHONPATH=/root/MaterialAgent python -m material_agent.cli \
  --partphys-scene /root/autodl-tmp/results_partphys/<scene_name> \
  --output-dir /root/autodl-tmp/results_partphys/<scene_name>/material_agent \
  --physgm-root /root/PhysGM \
  --partphys-root /root/PartPhysAgent \
  --physgm-config /root/PhysGM/configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --candidate-budget 3 \
  --backend auto \
  --simulate
```

With VLM material priors:

```bash
export DASHSCOPE_API_KEY=...
PYTHONPATH=/root/MaterialAgent python -m material_agent.cli \
  --partphys-scene /root/autodl-tmp/results_partphys/<scene_name> \
  --output-dir /root/autodl-tmp/results_partphys/<scene_name>/material_agent_vlm \
  --physgm-root /root/PhysGM \
  --partphys-root /root/PartPhysAgent \
  --physgm-config /root/PhysGM/configs/infer.yaml \
  --checkpoint /root/PhysGM/checkpoints/checkpoint.pt \
  --template-config /root/PhysGM/configs/physical/down_template.json \
  --config /root/MaterialAgent/configs/stable_solver.yaml \
  --candidate-budget 3 \
  --backend aabb \
  --vlm-provider openai_compatible \
  --vlm-model qwen3.7-plus \
  --vlm-api-base https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-key-env DASHSCOPE_API_KEY \
  --simulate \
  --white-bg
```

## Main Outputs

- `part_evidence.json`
- `distribution/part_distribution_outputs.json`
- `vlm_material/vlm_material_priors.json`
- `part_posteriors.json`
- `candidate_sets.json`
- `compiled_candidates.json`
- `simulation_results.json`
- `video_scores.json`
- `selected_materials.json`
- `sim_config_materialagent_selected.json`
- `selected_part_materials.json` when the selected backend is `part_id`
- `report.md`

## Verified Cake Run

The implementation was validated on the server with the cake example:

```text
/root/autodl-tmp/results_partphys/material_agent_full_cake_20260624_134745
```

The stable VLM run selected `soft_response` and produced:

```text
/root/autodl-tmp/results_partphys/material_agent_full_cake_20260624_134745/material_agent_full_vlm_aabb_stable/candidate_outputs/soft_response/output.mp4
```

Video validation used the rendered PNG frames because `ffmpeg/ffprobe` were not installed on the server:

- 50 frames
- 800 x 800 resolution
- nonblank rendered object in sampled frames
- measurable frame-to-frame motion
- minor black particle artifacts remain in the selected render

## Notes

`part_id` is preferred when per-particle material kernels are stable. For the cake VLM run, Ceramic plate priors
could trigger Warp CUDA illegal-address failures in candidate simulations, so the verified stable run uses the
`aabb` backend with conservative solver clamps while preserving the per-part visual material, raw `E`, and raw `nu`
predictions.
