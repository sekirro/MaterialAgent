# MaterialAgent

MaterialAgent is a part-level material-parameter agent for PhysGM workflows. It
loads a PartPhysAgent scene, gathers material evidence for each part, builds
posterior estimates for material, Young's modulus `E`, Poisson's ratio `nu`, and
density, generates candidate simulation configs, optionally runs candidate
simulations, scores the rendered outputs, and writes the selected config.

## Current Behavior

- Input is an existing PartPhysAgent scene directory.
- Material classes are the 14 PhysGM visual/material classes used by the model.
- Solver classes are mapped to PhysGM/MPM solver material branches.
- Per-part PhysGM crop inference is used as evidence when available.
- OpenAI-compatible VLM material priors are optional.
- Candidate configs are generated from posterior statistics and role-based
  candidate actions.
- If `--simulate` or `--mock-sim` is enabled, the controller runs a
  plan-act-observe-critique-repair loop.
- Backend `auto` chooses `part_id` when `assignment/gaussian_part_ids.npy`
  exists; otherwise it uses AABB `additional_material_params`.
- Multi-object PartPhysAgent scenes are supported when they expose merged
  per-object PhysGM geometry. The loader uses the actual `point_clouds.ply`
  parent directory as the simulation `--model_path`.
- Residual/unknown parts with assigned Gaussians are included in part-id material
  JSON so every non-fallback part id can receive explicit material parameters.
- `rigid_project` and `interface_bond` are candidate actions, not global behavior
  for every candidate.
- Solver stability can reduce `substep_dt` for stiff candidates according to the
  configured stability rule.

## Repository Layout

```text
material_agent/
  cli.py                         # command-line entry point
  loaders/partphys_outputs.py    # PartPhysAgent output loader
  evidence/physgm_distribution.py
  evidence/vlm_material.py
  reasoning/posterior.py
  reasoning/candidate_sampler.py
  reasoning/skill_memory.py
  simulation/config_compiler.py
  simulation/runner.py
  evaluation/video_metrics.py
  evaluation/critic.py
  evaluation/selector.py
configs/
  default.yaml
  stable_solver.yaml
tests/
  test_material_agent_core.py
```

## Command

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
  --config /root/MaterialAgent/configs/default.yaml \
  --candidate-budget 6 \
  --backend auto \
  --simulate \
  --white-bg
```

With OpenAI-compatible VLM priors:

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
  --candidate-budget 6 \
  --backend auto \
  --vlm-provider openai_compatible \
  --vlm-model qwen3.7-plus \
  --vlm-api-base https://llm-jrkem52i075alacx.cn-beijing.maas.aliyuncs.com/compatible-mode/v1 \
  --vlm-api-key-env DASHSCOPE_API_KEY \
  --simulate \
  --white-bg
```

## Evidence

The loader reads PartPhysAgent outputs including:

```text
partphys_summary.json
parts/
assignment/gaussian_part_ids.npy
assignment/per_part_aabb.json
physgm_whole/
```

For multi-object PartPhysAgent outputs, the point cloud may live under:

```text
physgm_whole/per_object_physgm/point_clouds.ply
```

In that case MaterialAgent uses `physgm_whole/per_object_physgm` as the PhysGM
model path for `part_id` simulations.

The evidence stage can use:

- PartPhysAgent part metadata and expected materials;
- per-part PhysGM crop predictions from `PhysGMDistributionExtractor`;
- optional VLM material priors from `VLMPartMaterialPriorExtractor`;
- optional skill memory from `material_skills.yaml`;
- whole-object PhysGM metadata as weak global evidence.

By default, PhysGM crop evidence uses the `context_dim` crop variant. Use:

```bash
--physgm-crop-variants all
```

to run all available crop variants.

## Candidate Generation

The current sampler can generate candidates with ids such as:

```text
posterior_map
support_stiff_mpm_response
solver_compatible_response
rigid_support_response
hard_support_bonded_response
soft_response
stiff_response
uncertain_part_sweep
alternative_material
```

Only candidates allowed by the scene evidence are produced. The final number is
limited by `--candidate-budget`.

Repair rounds can add candidates such as:

```text
repair_r<round>_support_stiffness
repair_r<round>_rigid_support
repair_r<round>_elastic_cohesive
repair_r<round>_higher_cohesion
repair_r<round>_solver_stability
repair_r<round>_role_balanced
```

Repair candidates are generated only when simulation is enabled, the selected
candidate fails the critic, and the round budget allows another round.

## Simulation Backends

`part_id` backend:

- writes per-part material JSON;
- runs PartPhysAgent's `tools/gs_simulation_partid_materials.py`;
- requires `assignment/gaussian_part_ids.npy`;
- supports merged per-object PhysGM outputs when the PartPhysAgent loader finds
  the corresponding merged `point_clouds.ply`.

`aabb` backend:

- writes PhysGM-compatible `additional_material_params`;
- runs `/root/PhysGM/gs_simulation.py`;
- uses per-part AABBs from PartPhysAgent assignment metadata.

`auto` chooses `part_id` when Gaussian part ids exist, otherwise `aabb`.

## Selection

When simulations are run, each candidate is compiled, executed, evaluated, and
passed to the selector. The critic can request repairs for support stiffness,
rigid support projection, deformable cohesion, solver stability, or role-balanced
parameters. The selected candidate and trace are written to disk.

Manual selection is available:

```bash
--selection human --select-candidate <candidate_id>
```

## Main Outputs

```text
part_evidence.json
agent_input_evidence.json
distribution/part_distribution_outputs.json
vlm_material/vlm_material_priors.json
part_posteriors.json
candidate_sets.json
compiled_candidates.json
candidate_configs/
candidate_outputs/
simulation_results.json
video_scores.json
selection.json
selected_materials.json
selected_compiled.json
sim_config_materialagent_selected.json
selected_part_materials.json
agent_trace.json
material_state.json
material_skill_memory_delta.yaml
report.md
```

`selected_part_materials.json` is written when the selected backend is `part_id`.

## Tests

```bash
python -m pytest tests/test_material_agent_core.py
```
