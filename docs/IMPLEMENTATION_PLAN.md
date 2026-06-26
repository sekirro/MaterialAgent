# MaterialAgent Implementation Plan

## Assumptions

1. The integration target is the GitHub version of `sekirro/PartPhysAgent`, commit `6174468424d73bc705e78b590f227c9964ec9d78`.
2. MaterialAgent will not train PhysGM or modify PhysGM weights in the MVP.
3. MaterialAgent will not redo part segmentation. It consumes PartPhysAgent's part schema, masks, multi-view masks, Gaussian-part assignment, AABBs, and whole-object PhysGM result.
4. The MVP keeps PhysGM's current solver contract: global constitutive material class plus local `E`, `nu`, and `density` through `additional_material_params`.
5. True per-particle material class switching is Phase 2, because it requires solver support beyond local scalar parameter injection.

## Success Criteria

For a multi-material object such as a hammer:

- MaterialAgent selects different material parameters for different parts, for example `head=Metal`, `handle=Wood` or `Plastic`.
- It generates a small candidate set of simulations, renders videos, and records a ranked decision.
- It writes a selected PhysGM config with part-local `additional_material_params`.
- It saves auditable evidence: candidate parameters, video paths, scores, selection reason, warnings, and skill-memory updates.
- It can run in a mock/test mode without a real GPU simulation.

## Input Contract

Recommended upstream command:

```bash
python partphys_pipeline.py \
  --image <image> \
  --scene-name <scene> \
  --output-dir <results_root> \
  --physgm-root <PhysGM> \
  --physgm-config configs/infer.yaml \
  --checkpoint <checkpoint.pt> \
  --template-config configs/physical/down_template.json \
  --segmentation-only
```

In the GitHub PartPhysAgent version inspected here, `--segmentation-only` still runs whole-object PhysGM and Gaussian assignment, but skips per-part PhysGM and simulation config. That is the cleanest handoff point for MaterialAgent.

Expected PartPhysAgent output directory:

```text
<scene>/
  input/input.png
  input/object_isolated_full.png
  schema/part_schema.json
  parts/part_XXX_<name>/mask.png
  parts/part_XXX_<name>/part_summary.json
  parts/parts_overlay.png
  assignment/gaussian_part_ids.npy
  assignment/assignment_summary.json
  assignment/part_gaussian_index.json
  assignment/per_part_aabb.json
  physgm_whole/point_clouds.ply
  physgm_whole/predicted_phys.json
  physgm_whole/input_batch_meta.npz
```

If `assignment/per_part_aabb.json` is missing, MaterialAgent should rebuild AABBs from `point_clouds.ply` plus `gaussian_part_ids.npy`.

## Output Contract

MaterialAgent writes:

```text
<scene>/material_agent/
  material_state.json
  part_evidence.json
  part_posteriors.json
  part_material_candidates.json
  candidate_sets.json
  candidate_configs/candidate_XXX.json
  candidate_videos/candidate_XXX.mp4
  candidate_keyframes/candidate_XXX.png
  video_scores.json
  selected_materials.json
  sim_config_materialagent_selected.json
  material_skill_memory_delta.yaml
  report.md
```

The selected materials file should keep the same semantic fields as PartPhysAgent's `PhysicsParams`:

```json
{
  "part_id": 0,
  "part_name": "head",
  "material": "Metal",
  "material_confidence": 0.83,
  "E": 2000000.0,
  "nu": 0.30,
  "density": 3000.0,
  "raw_E": 200000000000.0,
  "raw_nu": 0.30,
  "source": "video_selected_candidate_002",
  "warnings": []
}
```

`E/nu/density` are the solver-safe values used in simulation; `raw_E/raw_nu` preserve the physically interpreted values before solver clamp.

## Architecture

```text
MaterialAgent
  -> EvidenceLoader
  -> PartEvidenceBuilder
  -> PhysGMDistributionAdapter
  -> MaterialPosteriorBuilder
  -> CandidateSetSampler
  -> SimulationConfigCompiler
  -> SimulationRunner
  -> VideoEvaluator
  -> CandidateSelector
  -> SkillMemory
  -> ReportWriter
```

### 1. EvidenceLoader

Reads PartPhysAgent outputs and validates:

- part IDs are unique
- masks exist
- each non-residual part has a name and physics group
- whole-object PhysGM output exists
- Gaussian assignment has enough support per part
- AABB coordinate space is compatible with PhysGM's `additional_material_params`

Validation failure should stop before expensive simulation.

### 2. PartEvidenceBuilder

Collects evidence per part:

- part name, prompts, physical role, expected materials
- part crop images from PartPhysAgent or generated locally
- optional multi-view part masks
- material prior from VLM over the crop/context image
- existing PartPhysAgent crop PhysGM outputs if present
- whole-object global PhysGM material prior
- skill-memory matches from previous successful runs

The agent should treat VLM output as a prior, not ground truth.

### 3. PhysGMDistributionAdapter

The GitHub PartPhysAgent wrapper currently decodes only `E_mu` and `nu_mu`. PhysGM's model also returns `E_var` and `nu_var`.

Add an adapter that exposes:

```json
{
  "material_logits": [],
  "material_probs": {},
  "E_mu_norm": 0.0,
  "E_var_norm": 0.0,
  "nu_mu_norm": 0.0,
  "nu_var_norm": 0.0,
  "E_mu": 0.0,
  "E_sigma_log10": 0.0,
  "nu_mu": 0.0,
  "nu_sigma": 0.0
}
```

Decode samples using PhysGM's existing normalization:

```text
E_sample = 0.1 * 10 ** ((E_mu_norm + eps * sqrt(E_var_norm)) * E_STD + E_MEAN)
nu_sample = (nu_mu_norm + eps * sqrt(nu_var_norm)) * NU_STD + NU_MEAN
```

Then clamp samples to:

- material-specific physical ranges from PartPhysAgent's `material_table.py`
- solver-safe ranges before writing simulation config

If `E_var/nu_var` are unavailable, estimate uncertainty from the four crop variants and use a wider fallback distribution.

### 4. MaterialPosteriorBuilder

Build a posterior-like belief for each part:

```text
P(material | evidence)
P(log10(E), nu | material, evidence)
```

Evidence weights:

- PartPhysAgent schema `expected_materials`: medium-high
- PhysGM part crop material logits: high if crops agree
- VLM crop material prior: medium
- part name and physical role: medium
- skill memory: low to medium, never dominant alone
- whole-object PhysGM material: low for heterogeneous objects, medium for single-material objects

This module outputs top material hypotheses and continuous distributions for each part.

### 5. CandidateSetSampler

Avoid the full Cartesian product across parts. Use a budgeted sampler that creates 3 to 7 whole-object candidate sets:

1. `baseline_partphys`: current PartPhysAgent aggregate values.
2. `posterior_map`: highest posterior material and median `E/nu` per part.
3. `soft_response`: lower `E` quantile for deformable parts, higher `nu` for rubber/foam/plasticine.
4. `stiff_response`: upper `E` quantile for rigid/support/impact parts.
5. `uncertain_part_sweep`: vary only the lowest-confidence part while keeping others at MAP.
6. Optional `skill_memory_candidate`: values suggested by prior successful examples.
7. Optional human/VLM candidate requested by user.

Default budget: 5 candidate sets.

### 6. SimulationConfigCompiler

For every candidate set:

- choose a global base material from whole PhysGM or dominant/largest part
- preserve PartPhysAgent's `additional_material_params` schema
- write local `E`, `nu`, `density` per part AABB
- record raw and solver-clamped values separately
- reject invalid candidates before simulation

MVP should call or mirror PartPhysAgent's `build_part_aware_sim_config` behavior for compatibility.

### 7. SimulationRunner

Runs:

```bash
python gs_simulation.py \
  --model_path <physgm_whole> \
  --output_path <candidate_output> \
  --config <candidate_config> \
  --render_img \
  --compile_video
```

It must capture:

- command
- return code
- stdout/stderr
- video path
- keyframe contact sheet
- runtime
- simulation crash/NaN/explosion diagnostics

The runner supports `--mock-sim` for tests.

### 8. VideoEvaluator

Use a layered evaluator:

1. Hard filters:
   - simulation failed
   - video missing or blank
   - object explodes out of frame
   - too few visible frames

2. Physical sanity metrics:
   - total motion and acceleration smoothness
   - part-wise deformation ratio
   - rigid parts should deform less than soft parts
   - support parts should not jitter unrealistically
   - no severe interpenetration or collapse

3. Visual quality:
   - temporal smoothness
   - non-blank rendered frames
   - object remains visible

4. Optional VLM/human ranking:
   - create a candidate keyframe sheet
   - ask a VLM or user to choose the most physically plausible video
   - if user picks a candidate, accept immediately and store that preference

When no GT video exists, ranking is relative among candidates. If a GT or reference video exists later, add SAM2 plus CoTracker trajectory alignment, following PhysGM's DPO data-construction idea.

### 9. CandidateSelector

Selection policy:

- If the user manually chooses a video, select it.
- Else if VLM ranking is enabled and valid, combine VLM rank with hard metrics.
- Else use deterministic score.
- If all videos fail, fall back to `posterior_map` without simulation, mark low confidence, and save warnings.

The selector should emit a compact reason:

```json
{
  "selected_candidate": 2,
  "reason": "candidate has stable rigid head, moderate handle motion, no solver warnings, and best VLM rank",
  "score": 0.78
}
```

### 10. SkillMemory

Skill memory is a YAML/JSON database, not model training.

Store:

- object class
- part name / physics group
- visual material
- raw and solver-safe `E/nu/density`
- candidate score
- success/failure
- evaluator notes
- source scene

Retrieval keys:

- exact object + part
- part name only, such as `wheel`, `handle`, `blade`, `head`
- physical role, such as `impact part`, `support`, `soft coating`
- material label

Memory should influence priors, not override evidence.

## Proposed Project Layout

```text
MaterialAgent/
  README.md
  configs/default.yaml
  docs/
    IMPLEMENTATION_PLAN.md
    RESEARCH_NOTES.md
  material_agent/
    __init__.py
    cli.py
    state.py
    schemas.py
    loaders/
      partphys_outputs.py
    evidence/
      crops.py
      vlm_material_prior.py
      physgm_distribution.py
    reasoning/
      posterior.py
      candidate_sampler.py
      skill_memory.py
    simulation/
      config_compiler.py
      runner.py
    evaluation/
      video_metrics.py
      vlm_judge.py
      selector.py
    report.py
  tests/
    fixtures/
    test_distribution_decode.py
    test_candidate_sampler.py
    test_config_compiler.py
    test_skill_memory.py
```

## CLI Design

```bash
python -m material_agent.cli \
  --partphys-scene /path/to/results_partphys/hammer_partphys \
  --physgm-root /path/to/PhysGM \
  --template-config /path/to/down_template.json \
  --candidate-budget 5 \
  --simulate \
  --selection auto
```

Modes:

- `--selection auto`: deterministic video metrics plus optional VLM.
- `--selection human`: render candidates and write a selection sheet, then wait for a user-provided candidate ID in a follow-up command.
- `--selection vlm`: use a VLM to rank candidate keyframe sheets.
- `--mock-sim`: skip GPU simulation for tests.

## Implementation Milestones

### Stage 0: Project skeleton

Deliverables:

- package layout
- schemas
- config loader
- test fixtures

Verify:

- `python -m pytest MaterialAgent/tests`
- CLI help works

### Stage 1: Read PartPhysAgent outputs

Deliverables:

- `EvidenceLoader`
- `PartEvidenceBuilder`
- AABB rebuild fallback

Verify:

- loads a mock scene
- rejects missing masks/AABBs with clear errors

### Stage 2: PhysGM distribution adapter

Deliverables:

- expose `E_mu/E_var/nu_mu/nu_var`
- decode parameter samples
- material table clamp

Verify:

- deterministic decode matches current PartPhysAgent mean path
- samples stay in valid ranges

### Stage 3: Candidate sampler

Deliverables:

- posterior builder
- budgeted candidate-set sampler
- candidate JSON output

Verify:

- hammer produces at least two different part material sets
- no Cartesian explosion as part count grows

### Stage 4: Simulation compiler and runner

Deliverables:

- per-candidate config files
- simulation runner
- mock runner

Verify:

- config contains valid `additional_material_params`
- mock candidate videos are registered

### Stage 5: Video evaluator and selector

Deliverables:

- hard failure filters
- keyframe sheet generator
- deterministic scoring
- optional VLM/human selection

Verify:

- failed simulations are never selected
- manual candidate selection produces selected config

### Stage 6: Skill memory

Deliverables:

- memory retrieval
- memory update from winner/failure
- confidence-aware prior blending

Verify:

- repeated hammer-like scenes retrieve handle/head priors
- memory cannot override strong contradictory evidence alone

## Evaluation Plan

Baselines:

1. PhysGM global material for all parts.
2. PartPhysAgent current crop aggregation.
3. MaterialAgent posterior MAP without video selection.
4. MaterialAgent with candidate video selection.
5. Human-selected upper bound.

Metrics:

- material label agreement on curated multi-material examples
- solver success rate
- video physical plausibility score
- part-wise deformation consistency
- runtime and number of simulations
- user/VLM preference win rate

Example objects:

- hammer: metal head, wood/plastic handle
- shoe: rubber sole, fabric/leather upper, laces
- cake: foam cake body, frosting/cream, plate
- chair: rigid frame, cushion
- toy car: plastic body, rubber wheels, metal axles if visible

## Main Risks

- PhysGM variance may be poorly calibrated. Mitigation: combine distribution samples with crop disagreement and material-table priors.
- VLM material priors may hallucinate. Mitigation: candidate-based selection and hard clamps.
- Simulation is expensive. Mitigation: small candidate budget and early hard filters.
- `additional_material_params` uses AABBs, so adjacent parts can overlap. Mitigation: prefer per-particle material params in Phase 2.
- Current solver keeps one global constitutive material class. Mitigation: Phase 2 adds per-particle material IDs or solver material mixtures.

## Phase 2: True Per-Particle Materials

After MVP:

- write `per_particle_material_params.npz`
- assign `particle_E`, `particle_nu`, `particle_density`, and `particle_material_id` by Gaussian/particle part labels
- modify or reuse solver hooks that support per-particle material class
- compare AABB local params vs exact particle labels

This is the path toward OmniPhysGS-like heterogeneous material behavior while keeping PartPhysAgent's part decomposition.
