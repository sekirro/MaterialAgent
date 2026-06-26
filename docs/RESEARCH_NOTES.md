# Research Notes

These notes summarize the papers and systems that shape MaterialAgent.

## PhysGM

[PhysGM](https://arxiv.org/abs/2508.13911) predicts a 3D Gaussian representation and physical properties from an image in a feed-forward pass. The important detail for this project is its probabilistic physics head: it predicts material class plus distributions for continuous parameters such as Young's modulus and Poisson's ratio. Its supplementary candidate-video process samples multiple physical parameter sets, simulates each one, and ranks videos by trajectory alignment. Its stated limitation is exactly our target: it predicts one lumped physical-property vector for the whole object, assuming uniform material composition.

Implementation implication:

- Reuse the PhysGM distribution head for parameter uncertainty.
- Do not keep one global object parameter when PartPhysAgent has part masks and 3DGS part labels.
- Use PhysGM's candidate-video ranking idea at inference time, but initially as a test-time selection loop rather than model fine-tuning.

## PartPhysAgent

The target integration is [sekirro/PartPhysAgent](https://github.com/sekirro/PartPhysAgent), commit `6174468424d73bc705e78b590f227c9964ec9d78`.

Key interfaces observed in that GitHub version:

- `partphys_pipeline.py` exposes `--mask-only`, `--segmentation-only`, `--whole-physgm-dir`, `--agent-mode`, `--multiview-dir`, `--skip-part-physgm`, and PhysGM paths.
- `README_PARTPHYS.md` states that whole-object PhysGM is the final geometry source, part-crop PhysGM estimates per-part physical parameters, and local parameters are injected through `additional_material_params`.
- `partphys/types.py` defines `PartInstance`, `PhysicsParams`, `PhysGMResult`, and `PartPhysResult`.
- `partphys/agent.py` currently generates four crops per part, runs PhysGM on each crop, and aggregates material/E/nu with a weighted median.
- `partphys/physgm_runner.py` currently decodes `E_mu` and `nu_mu`, but does not export `E_var` and `nu_var`, even though the PhysGM model returns them.
- `partphys/sim_config_builder.py` writes local `E`, `nu`, and `density` into `additional_material_params`, with solver-stability clamps.
- `partphys/gaussian_assign.py` supports multi-view projection voting, low-confidence smoothing, and KNN cleanup for part labels.

Implementation implication:

- MaterialAgent can be a downstream consumer of PartPhysAgent outputs rather than a fork.
- The first code change needed around PhysGM is a small adapter that exposes `E_var` and `nu_var` from the model output.
- MaterialAgent should preserve the existing `PhysicsParams` shape for compatibility.

## Related Work

[PhysGaussian](https://arxiv.org/abs/2311.12198) couples 3DGS and MPM, establishing the "what you see is what you simulate" style pipeline, but uses manually configured material properties. MaterialAgent automates that parameter choice.

[PhysDreamer](https://arxiv.org/abs/2404.13026), [DreamPhysics](https://arxiv.org/abs/2406.01476), and [Physics3D](https://arxiv.org/abs/2406.04338) use video diffusion priors or distillation to infer physical behavior. They motivate video-based feedback, but their per-scene optimization cost is too high for our first version.

[GaussianProperty](https://arxiv.org/abs/2412.11258) combines segmentation, multimodal material reasoning, and multi-view projection voting to assign physical properties to 3D Gaussians. This is close to MaterialAgent's evidence-gathering side.

[OmniPhysGS](https://arxiv.org/abs/2501.18982) argues that single-category material assumptions fail for heterogeneous objects and represents each Gaussian with material expert mixtures. MaterialAgent starts coarser, at part level, but should leave a path to per-particle or per-Gaussian materials.

[PhysGS](https://arxiv.org/abs/2511.18570) models dense physical properties with Bayesian uncertainty over Gaussian splats. This supports MaterialAgent's posterior-style design: keep beliefs and uncertainty, then choose candidates under a budget.

[PhysX-3D](https://arxiv.org/abs/2507.12465) builds physics-grounded 3D asset data with scale/material/affordance/kinematics/function annotations. This is useful future supervision for the skill memory or a learned prior.

## Design Takeaways

- Part-level material reasoning is the right middle ground: finer than PhysGM's whole-object vector, cheaper than OmniPhysGS-style per-Gaussian expert mixtures.
- Use a posterior over material/E/nu, not a single direct guess.
- Keep candidate counts small. Simulations are expensive, so the default should be 3 to 7 candidate sets.
- Video selection is useful even without ground-truth videos: use automatic physical sanity metrics plus optional VLM or human choice.
- Save reusable decisions as skill memory, but never silently override visual evidence with memory.

