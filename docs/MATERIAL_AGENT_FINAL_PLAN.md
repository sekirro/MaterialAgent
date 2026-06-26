# MaterialAgent 最终详细方案

## 0. 一句话目标

MaterialAgent 是接在 `sekirro/PartPhysAgent` 后面的 part-level material reasoning agent。它不重新做分割，不训练 PhysGM，而是读取 PartPhysAgent 已经得到的物理 part、Gaussian assignment 和 whole-object PhysGM 输出，为每个 part 推断：

- `material`: 视觉/语义材料类别，例如 `Metal`, `Wood`, `Rubber`, `Foam`
- `E`: Young's modulus
- `nu`: Poisson's ratio
- `density`: 密度

然后它采样少量候选材料参数组合，分别运行 PhysGM/MPM 仿真并渲染视频，通过自动指标、VLM 或人工选择最合理的视频，最后输出最优 part-aware simulation config，并把成功/失败经验沉淀成 material skill memory。

## 1. 当前依据和假设

### 1.1 PartPhysAgent 依据

目标上游仓库是：

- GitHub: https://github.com/sekirro/PartPhysAgent
- 当前 main commit: `6174468424d73bc705e78b590f227c9964ec9d78`

这个版本已经提供：

- object mask
- part schema
- part masks
- optional multi-view part masks
- whole-object PhysGM output
- Gaussian-to-part assignment
- per-part AABB
- `additional_material_params` config 写入路径

最适合作为 MaterialAgent 输入的上游命令是：

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

原因：这个 GitHub 版本的 `--segmentation-only` 仍然会运行 whole-object PhysGM 和 Gaussian assignment，但会跳过 per-part PhysGM 与 simulation config。也就是说，MaterialAgent 可以在这里接管“材料参数选择”。

### 1.2 PhysGM 实现依据

本地可读的原始 PhysGM 实现在：

```text
PhysGM-git/
```

关键代码事实：

1. `model/physgm.py` 中，三个 global token 分别用于材料类别、`E` 和 `nu`：

```text
phys_tokens = aggregated_tokens[-1][:, 0]
E_tokens    = aggregated_tokens[-1][:, 1]
nu_tokens   = aggregated_tokens[-1][:, 2]
```

2. 物理头实际输出均值和方差：

```text
phys_logits = phys_token_decoder(phys_tokens)
E_logits = E_token_decoder(E_tokens)
E_mu  = E_logits[..., 0]
E_var = softplus(E_logits[..., 1]) + 1e-2
nu_logits = nu_token_decoder(nu_tokens)
nu_mu  = nu_logits[..., 0]
nu_var = softplus(nu_logits[..., 1]) + 1e-2
```

3. `ret_dict` 确实返回：

```text
E_mu, E_var, nu_mu, nu_var, phys_logits
```

4. 训练损失对 `E` 和 `nu` 使用 Gaussian NLL，并额外加 MSE：

```text
NLL(mu, var, target) + MSE(mu, target)
```

所以 PhysGM 的分布存在，但当前 inference pipeline 没有保存出来。

5. `pipeline.py` 只把均值解码为 `predicted_phys.json`：

```text
E = 0.1 * 10 ** (E_mu_norm * E_STD + E_MEAN)
nu = nu_mu_norm * NU_STD + NU_MEAN
material = argmax(phys_logits)
```

6. 归一化常数是：

```text
E_MEAN = 7.387210
E_STD  = 2.456477
NU_MEAN = 0.398
NU_STD  = 0.111
```

7. `gs_simulation.py` 通过 `decode_param_json()` 读配置，然后调用：

```text
mpm_solver.set_parameters_dict(material_params)
mpm_solver.finalize_mu_lam()
```

8. 原始 solver 的 `additional_material_params` 只会按 AABB 改局部粒子的：

```text
E, nu, density
```

它不会改变每个 part 的 constitutive material class。`model.material` 仍是全局单一值，控制 stress/return mapping 走 `jelly/metal/sand/foam/snow/plasticine` 哪个分支。

这个点非常重要：MVP 的 MaterialAgent 可以做到 per-part `E/nu/density`，但不能真正做到每个 part 一个 solver material class。真正 per-part constitutive class 是 Phase 2。

## 2. 文献调研结论

### 2.1 基础表示和物理仿真

[3D Gaussian Splatting](https://arxiv.org/abs/2308.04079) 证明了 3D Gaussian 是高效、显式、可实时渲染的 3D 表示。MaterialAgent 继承这个表示，不再引入 mesh 或 tetrahedral embedding。

[PhysGaussian](https://arxiv.org/abs/2311.12198) 提出 3DGS 与 MPM 直接耦合，核心思想是 "what you see is what you simulate"。它证明 Gaussian 可以作为仿真/渲染的共同离散载体，但材料参数需要手工或 per-scene 调整。

MPM 相关工作说明 Material Point Method 适合大变形、弹塑性、颗粒、雪、泡沫等 continuum materials。PhysGM 的 solver 正是基于这个方向。

### 2.2 用视频/生成模型估计物理属性

[PhysDreamer](https://arxiv.org/abs/2404.13026)、[DreamPhysics](https://arxiv.org/abs/2406.01476)、[Physics3D](https://arxiv.org/abs/2406.04338) 都在探索用视频扩散模型或视频先验来反推物理属性。它们的共同启发是：

- 单张图很难直接确定真实材料参数。
- 用候选仿真视频和视觉反馈选择参数更稳。
- 但逐场景优化成本高，不适合作为 MaterialAgent MVP 的主路径。

因此 MaterialAgent 采用 test-time candidate selection，而不是 SDS 优化。

### 2.3 Gaussian 上的物理属性推断

[GIC](https://arxiv.org/abs/2406.14927) 用 Gaussian-informed continuum 做视觉观测下的物理属性识别，说明 Gaussian 表示能作为物理识别和仿真的桥梁。

[GaussianProperty](https://arxiv.org/abs/2412.11258) 使用 SAM、GPT-4V、多视图投票给 3D Gaussians 标注物理属性。它对 MaterialAgent 的启发是：

- 材料判断应该使用 global-local reasoning。
- 多视图 2D evidence 可以投票到 3D Gaussian。
- 训练 free 的 LMM/VLM 物理属性推理可以作为 prior，但需要几何一致性约束。

[PhysGS](https://arxiv.org/abs/2511.18570) 进一步强调 Bayesian physical property estimation 和不确定性建模。MaterialAgent 的 posterior 设计正是借这个思想：不要只输出一个值，而要维护 material/E/nu 的置信度和不确定性。

### 2.4 多材料/异质物体

[OmniPhysGS](https://arxiv.org/abs/2501.18982) 明确指出单一预设材料类别难以覆盖真实复合物体，并提出每个 Gaussian 由多个 physical domain expert 组合。这个方向很强，但实现复杂。

MaterialAgent 取中间层：

```text
whole-object material < part-level material < per-Gaussian material mixture
```

也就是先做 part-level posterior 和候选视频选择，之后再扩展到 per-particle material class。

### 2.5 PhysGM 本身

[PhysGM](https://arxiv.org/abs/2508.13911) 的关键贡献是 feed-forward 同时预测 3DGS 和物理属性，并用 DPO 对仿真视频偏好进行对齐。它的补充材料里有一个重要流程：

```text
从物理参数分布中采样多个 E/nu
  -> 分别运行 MPM 仿真
  -> 渲染多个候选视频
  -> 用轨迹对齐/偏好选择 winner
```

MaterialAgent 直接把这个思路迁移到 inference-time 的 part-level 参数选择上。

PhysGM 的局限也正好是本项目贡献点：它目前预测整个物体一个 lumped physical vector，假设材料均匀。真实物体如 hammer 有 metal head 和 wooden/plastic handle，这个假设不成立。

### 2.6 辅助视觉工具

[SAM 2](https://arxiv.org/abs/2408.00714) 和 [CoTracker3](https://arxiv.org/abs/2410.11831) 可用于候选视频评价：

- SAM2: 分割视频中的物体/part 区域。
- CoTracker3: 跟踪点轨迹，用于比较候选视频的运动稳定性、落地后变形、抖动和漂移。

MVP 不强依赖它们，但 VideoEvaluator 的高级版本应该接入。

## 3. 总体设计

### 3.1 Agent 状态

MaterialAgent 维护一个显式状态：

```json
{
  "scene": "hammer_001",
  "input_partphys_scene": "...",
  "parts": [],
  "evidence": {},
  "posteriors": {},
  "candidate_sets": [],
  "simulation_results": [],
  "video_scores": [],
  "selection": {},
  "skill_memory_updates": [],
  "warnings": []
}
```

落盘路径：

```text
<scene>/material_agent/material_state.json
```

### 3.2 模块图

```text
PartPhysAgent outputs
  -> EvidenceLoader
  -> PartEvidenceBuilder
  -> PhysGMDistributionExtractor
  -> MaterialPosteriorBuilder
  -> CandidateSetSampler
  -> SimulationConfigCompiler
  -> SimulationRunner
  -> VideoEvaluator
  -> CandidateSelector
  -> SkillMemory
  -> ReportWriter
```

## 4. 输入输出协议

### 4.1 输入

MaterialAgent 读取：

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

最低要求：

- part schema 存在
- 每个非 residual part 有 mask
- whole-object `point_clouds.ply` 存在
- `gaussian_part_ids.npy` 或 `per_part_aabb.json` 至少存在一个

### 4.2 输出

MaterialAgent 写入：

```text
<scene>/material_agent/
  material_state.json
  part_evidence.json
  part_posteriors.json
  part_material_candidates.json
  candidate_sets.json
  candidate_configs/candidate_000.json
  candidate_configs/candidate_001.json
  candidate_outputs/candidate_000/
  candidate_outputs/candidate_001/
  candidate_keyframes/candidate_000.png
  video_scores.json
  selected_materials.json
  sim_config_materialagent_selected.json
  material_skill_memory_delta.yaml
  report.md
```

最终 `selected_materials.json`：

```json
{
  "parts": [
    {
      "part_id": 0,
      "part_name": "head",
      "physics_group": "head",
      "visual_material": "Metal",
      "solver_material": "global",
      "raw_E": 200000000000.0,
      "raw_nu": 0.30,
      "raw_density": 7800.0,
      "simulation_E": 2000000.0,
      "simulation_nu": 0.30,
      "simulation_density": 3000.0,
      "confidence": 0.84,
      "source": "selected_candidate_002",
      "warnings": []
    }
  ]
}
```

注意：`raw_*` 是物理意义上的参数；`simulation_*` 是经过 solver stability clamp 后实际写入 PhysGM config 的参数。

## 5. EvidenceLoader

EvidenceLoader 只做确定性读取和验证。

### 5.1 读取对象

```python
SceneEvidence:
    scene_dir
    input_image
    object_image
    part_schema
    parts: list[PartEvidence]
    whole_physgm
    gaussian_assignment
    part_aabbs
```

每个 part：

```python
PartEvidence:
    part_id
    name
    physics_group
    mask_path
    bbox_2d
    area
    expected_materials
    physical_role
    view_masks
    gaussian_count
    aabb_center
    aabb_half_size
    existing_partphys_physics
```

### 5.2 质量检查

必须检查：

- part id 是否唯一
- mask 是否存在
- AABB 是否有正 `half_size`
- Gaussian count 是否过低
- assignment ratio 是否太低
- residual/unknown part 是否应该参与材料选择

建议默认跳过：

```text
unknown_body
residual
background-like part
gaussian_count < 20
```

这些区域可以继承最近邻或 dominant part 的材料，不作为主要候选维度。

## 6. PartEvidenceBuilder

PartEvidenceBuilder 为每个 part 建证据包。

### 6.1 图像证据

生成或复用四种 crop：

```text
crop_tight.png
crop_padded.png
crop_context_dim.png
crop_isolated_full.png
```

这些和 PartPhysAgent 当前思路兼容。

### 6.2 语义证据

来自：

- `part_schema.json`
- part name
- text prompts
- `expected_materials`
- `physical_role`
- object name

例如 hammer：

```text
head: impact part, expected Metal
handle: grip/support part, expected Wood/Plastic
```

### 6.3 PhysGM crop 证据

如果 PartPhysAgent 已经跑过 per-part PhysGM，就读取其 `part_summary.json` 里的 aggregated physics 和 source outputs。

如果是推荐的 `--segmentation-only` 输入，MaterialAgent 自己运行 PhysGM distribution extraction。

### 6.4 VLM 证据

VLM 不直接决定最终材料，只输出 prior：

```json
{
  "part": "handle",
  "candidates": [
    {"material": "Wood", "confidence": 0.55, "reason": "brown texture and handle role"},
    {"material": "Plastic", "confidence": 0.35, "reason": "smooth manufactured grip"}
  ]
}
```

## 7. PhysGMDistributionExtractor

这是 MaterialAgent 最重要的新增模块。

### 7.1 为什么需要它

PhysGM 模型实际有分布：

```text
E_mu, E_var, nu_mu, nu_var
```

但当前 inference 输出只有：

```text
E, nu, material
```

MaterialAgent 要使用 PhysGM 的概率思想，就必须把 var 取出来。

### 7.2 解码公式

沿用 PhysGM 当前均值解码：

```text
E_mean = 0.1 * 10 ** (E_mu_norm * E_STD + E_MEAN)
nu_mean = nu_mu_norm * NU_STD + NU_MEAN
```

采样公式：

```text
z ~ N(0, 1)
E_norm_sample = E_mu_norm + z * sqrt(E_var_norm)
nu_norm_sample = nu_mu_norm + z * sqrt(nu_var_norm)

E_sample = 0.1 * 10 ** (E_norm_sample * E_STD + E_MEAN)
nu_sample = nu_norm_sample * NU_STD + NU_MEAN
```

然后 clamp：

```text
nu in [0.01, 0.49]
E > 0
material-specific range
solver-safe range
```

### 7.3 输出

每个 crop 输出：

```json
{
  "variant": "context_dim",
  "material_probs": {
    "Metal": 0.72,
    "Wood": 0.18,
    "Plastic": 0.05
  },
  "E_mu_norm": 0.12,
  "E_var_norm": 0.28,
  "nu_mu_norm": -0.32,
  "nu_var_norm": 0.14,
  "E_mean": 2000000000.0,
  "E_sigma_log10": 1.30,
  "nu_mean": 0.36,
  "nu_sigma": 0.041
}
```

### 7.4 方差校准

PhysGM 的 `E_var/nu_var` 不一定校准好，所以 MaterialAgent 不应该盲目相信单次 var。建议组合三种不确定性：

```text
model_uncertainty = sqrt(E_var/nu_var)
crop_disagreement = four crop outputs 的离散程度
semantic_ambiguity = material posterior entropy
```

最终采样宽度：

```text
sigma_final = w1 * model_sigma + w2 * crop_std + w3 * semantic_entropy_scale
```

MVP 默认：

```text
w1 = 0.5
w2 = 0.35
w3 = 0.15
```

## 8. MaterialPosteriorBuilder

目标是构建：

```text
P(material | evidence)
P(log10(E), nu | material, evidence)
```

### 8.1 material posterior

证据权重建议：

```yaml
schema_expected_material: 0.25
physgm_crop_logits: 0.35
vlm_crop_prior: 0.20
part_name_role_prior: 0.10
skill_memory_prior: 0.10
whole_object_prior: 0.05
```

如果是明显多 part 物体，`whole_object_prior` 权重降低。

### 8.2 E/nu posterior

用 `log10(E)` 建模，而不是直接对 `E` 做高斯。

每个 material 有物理范围：

```text
Metal: high E, nu around 0.20-0.35
Wood: high-but-lower E, nu around 0.25-0.45
Rubber: lower E, high nu
Foam/Fabric: low E, lower/medium nu
```

最终 posterior 不是纯 PhysGM 输出，而是：

```text
PhysGM distribution
  + material table range
  + part role prior
  + crop consistency
  + skill memory
```

### 8.3 part role prior

根据 `physical_role/name` 加先验：

```text
impact/head/blade/tip -> prefer rigid, high E
handle/grip -> Wood/Plastic/Rubber, medium E
sole/tire/wheel -> Rubber, high nu
cushion/padding/foam -> Foam/Fabric, low E
frosting/cream -> Foam/Plasticine, low E
plate/support -> Ceramic/Glass/Plastic, high E
```

## 9. CandidateSetSampler

不能对每个 part 的 top-k material 全排列，否则候选爆炸。

### 9.1 候选预算

默认：

```yaml
candidate_budget: 5
max_candidate_budget: 7
```

### 9.2 候选集合

生成整物体 candidate set，而不是单独 part candidate。

必选：

1. `baseline_partphys`
   - 如果 PartPhysAgent 已有 per-part 聚合结果，就直接使用。
   - 如果没有，就使用 PhysGM part crop posterior median。

2. `posterior_map`
   - 每个 part 取最高 posterior material 和 median E/nu。

3. `soft_response`
   - 对 soft/deformable part 取低 E 分位数。
   - 对 Rubber/Foam/Plasticine 取较高 nu。

4. `stiff_response`
   - 对 rigid/support/impact part 取高 E 分位数。
   - 用于避免头部/支撑件过软。

5. `uncertain_part_sweep`
   - 只改变最低置信 part，其它 part 固定为 MAP。

可选：

6. `skill_memory_candidate`
   - 使用历史成功经验。

7. `user_candidate`
   - 用户指定的 material/E/nu。

### 9.3 示例

Hammer:

```json
[
  {
    "candidate_id": "posterior_map",
    "parts": {
      "head": {"material": "Metal", "E_quantile": 0.50, "nu_quantile": 0.50},
      "handle": {"material": "Wood", "E_quantile": 0.50, "nu_quantile": 0.50}
    }
  },
  {
    "candidate_id": "handle_plastic_sweep",
    "parts": {
      "head": {"material": "Metal", "E_quantile": 0.50},
      "handle": {"material": "Plastic", "E_quantile": 0.40}
    }
  },
  {
    "candidate_id": "soft_handle",
    "parts": {
      "head": {"material": "Metal", "E_quantile": 0.70},
      "handle": {"material": "Wood", "E_quantile": 0.25}
    }
  }
]
```

## 10. SimulationConfigCompiler

### 10.1 MVP 写法

沿用 PhysGM/PartPhysAgent 当前接口：

```json
"additional_material_params": [
  {
    "point": [x, y, z],
    "size": [sx, sy, sz],
    "E": 2000000.0,
    "nu": 0.30,
    "density": 3000.0
  }
]
```

### 10.2 全局 material class

由于原始 solver 只有一个 `model.material`，MVP 需要选一个全局 solver material：

策略：

1. 如果 object 主要是 soft body，选 `jelly/foam/plasticine`。
2. 如果 object 主要是 rigid object，选 `plasticine` 或稳定的 rigid-like material。
3. 如果场景有 granular/snow，选对应 `sand/snow`。
4. 对 hammer 这类复合刚体，建议先用 solver-stable global class，再通过 local E/nu 体现刚度差异。

这是近似，不是完美多材料 constitutive simulation。

### 10.3 solver-safe clamp

原始物理参数可能非常大，例如真实 metal E 可到 `2e11 Pa`。但当前 Warp MPM solver 直接使用这个范围可能不稳定。

所以需要两套值：

```text
raw_E/raw_nu/raw_density: 物理解释值
simulation_E/simulation_nu/simulation_density: 实际仿真值
```

默认 simulation clamp：

```yaml
local_E_range: [1.0e3, 2.0e6]
local_nu_range: [0.05, 0.45]
local_density_range: [50.0, 3000.0]
```

这些范围可以按项目实验继续调。

## 11. SimulationRunner

每个 candidate 运行：

```bash
python gs_simulation.py \
  --model_path <physgm_whole> \
  --output_path <material_agent/candidate_outputs/candidate_XXX> \
  --config <material_agent/candidate_configs/candidate_XXX.json> \
  --render_img \
  --compile_video \
  --white_bg
```

记录：

```json
{
  "candidate_id": "candidate_002",
  "returncode": 0,
  "video_path": ".../video.mp4",
  "keyframes_path": ".../candidate_002.png",
  "stdout": ".../stdout.txt",
  "stderr": ".../stderr.txt",
  "runtime_sec": 83.2,
  "warnings": []
}
```

如果 `--mock-sim`：

- 不跑真实 GPU 仿真。
- 生成 mock video metadata。
- 用于单元测试 CandidateSampler/Selector。

## 12. VideoEvaluator

MaterialAgent 的关键贡献不是“猜一个 E/nu”，而是“候选视频闭环选择”。

### 12.1 Hard filters

候选直接失败的情况：

- 仿真 return code 非 0
- 没有视频
- 视频空白
- object 出画面
- 物体爆炸式散开
- 渲染帧数太少
- `stderr` 中出现 NaN/inf/solver error

### 12.2 无 GT 视频的自动指标

没有参考视频时，用相对评价：

```text
visibility_score
temporal_smoothness_score
motion_reasonableness_score
deformation_role_consistency_score
part_separation_stability_score
render_quality_score
```

具体：

- rigid part 的形变应小于 soft part。
- support/plate/base 不应剧烈抖动。
- rubber/foam/frosting 应允许更多形变。
- metal/head/blade 不应像软胶一样塌陷。
- object 不应整体弹飞或穿地。
- 运动轨迹应连续，不应 frame-to-frame 闪烁。

### 12.3 有生成参考视频时

可选生成一个 reference video prompt，例如 PhysGM 补充材料思路：

```text
An object made of <material description> falls straight down ...
```

但对 multi-material object 更推荐：

```text
A hammer with a metal head and wooden handle falls ...
```

然后用：

- SAM2 segmentation
- CoTracker3 trajectories
- bounding-box normalized trajectory distance

计算候选仿真和参考视频的相似度。

这适合高质量实验，不一定作为默认路径。

### 12.4 VLM/人工选择

最稳的 MVP 可以提供候选视频 contact sheet：

```text
candidate_000 | candidate_001 | candidate_002 | candidate_003
```

选择模式：

```yaml
selection: auto | vlm | human
```

- `auto`: 用 deterministic metrics。
- `vlm`: VLM 看 keyframes/video sheet，选择最物理合理的候选。
- `human`: 用户直接选候选编号，选完即结束。

人工选择后记录为 high-confidence skill memory。

## 13. CandidateSelector

综合分：

```text
score =
  0.25 * hard_valid
  + 0.20 * visual_quality
  + 0.20 * temporal_smoothness
  + 0.20 * physical_role_consistency
  + 0.10 * material_prior_consistency
  + 0.05 * solver_stability
```

如果 VLM/human 有选择：

- human 选择优先级最高。
- VLM 选择需要通过 hard filters。
- 自动指标用于 tie-break 和 reject bad VLM choice。

输出：

```json
{
  "selected_candidate": "candidate_002",
  "selection_mode": "auto",
  "score": 0.78,
  "reason": [
    "rigid head remains stable",
    "handle motion is plausible",
    "no explosion or blank frames",
    "best physical role consistency"
  ]
}
```

## 14. SkillMemory

MaterialAgent 的 skill 不是 prompt，而是结构化经验库。

### 14.1 存储格式

```yaml
entries:
  - key:
      object: hammer
      part_name: head
      physical_role: impact part
    material: Metal
    raw_E: 2.0e11
    raw_nu: 0.30
    simulation_E: 2.0e6
    simulation_nu: 0.30
    density: 3000.0
    source_scene: hammer_001
    score: 0.82
    outcome: success
    notes:
      - metal head should stay rigid
      - high raw E must be solver-clamped
```

### 14.2 检索方式

按优先级：

1. object + part name 精确匹配
2. part name 匹配，例如 `wheel`, `handle`, `blade`, `head`
3. physical role 匹配，例如 `impact part`, `support`, `soft coating`
4. material label 匹配

### 14.3 使用原则

Memory 只能作为 prior，不能覆盖视觉证据。

例如：

```text
历史上 hammer handle 多为 Wood，
但当前 crop/VLM/PhysGM 都强烈认为是 Plastic，
则 posterior 应保留 Plastic 候选，而不是强行 Wood。
```

## 15. 项目结构

```text
MaterialAgent/
  README.md
  configs/
    default.yaml
  docs/
    MATERIAL_AGENT_FINAL_PLAN.md
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
    test_partphys_loader.py
    test_distribution_decode.py
    test_posterior.py
    test_candidate_sampler.py
    test_config_compiler.py
    test_selector.py
    test_skill_memory.py
```

## 16. CLI 设计

```bash
python -m material_agent.cli \
  --partphys-scene <results_root>/<scene> \
  --physgm-root <PhysGM-git> \
  --template-config <PhysGM-git>/configs/physical/down_template.json \
  --candidate-budget 5 \
  --selection auto \
  --simulate
```

调试：

```bash
python -m material_agent.cli \
  --partphys-scene <scene> \
  --mock-sim \
  --candidate-budget 5 \
  --selection auto
```

人工选择：

```bash
python -m material_agent.cli \
  --partphys-scene <scene> \
  --candidate-budget 5 \
  --selection human \
  --simulate
```

然后：

```bash
python -m material_agent.cli \
  --partphys-scene <scene> \
  --select-candidate candidate_002
```

## 17. 实现阶段

### Stage 0: 文档和 schema

已完成：

- `MaterialAgent/README.md`
- `configs/default.yaml`
- 初版 implementation plan
- 本最终方案文档

验收：

- 文件存在。
- 文档明确输入/输出/阶段。

### Stage 1: Loader + schema

实现：

- `SceneEvidence`
- `PartEvidence`
- `EvidenceLoader`
- AABB fallback

验收：

- mock scene 可读。
- 缺失 mask/AABB 会明确报错。
- residual part 可跳过。

### Stage 2: PhysGM distribution extraction

实现：

- `PhysGMDistributionExtractor`
- 读取 `E_mu/E_var/nu_mu/nu_var/phys_logits`
- 解码 mean 和 samples
- material probability softmax

验收：

- 均值解码与原始 PhysGM `pipeline.py` 一致。
- 样本满足 material physical range。
- 输出 `part_distribution_outputs.json`。

### Stage 3: Posterior + candidate sampler

实现：

- material posterior
- logE/nu posterior
- 3-7 个 candidate set

验收：

- hammer 会生成 head/handle 不同材料候选。
- candidate 数不随 part 数指数爆炸。

### Stage 4: Config compiler + mock sim

实现：

- 写 candidate configs
- 写 selected config
- mock simulation runner

验收：

- `additional_material_params` 正确。
- raw/simulation 参数都保存。

### Stage 5: 真仿真 + video evaluator

实现：

- 调用 `gs_simulation.py`
- 保存 stdout/stderr/video/keyframes
- hard filters
- deterministic scoring

验收：

- crash/blank/explosion 不会被选中。
- 正常候选能进入 ranking。

### Stage 6: VLM/human selection

实现：

- keyframe sheet
- VLM judge prompt
- human select candidate

验收：

- 人工选 candidate 后直接生成 selected config。
- VLM 选择必须通过 hard filter。

### Stage 7: Skill memory

实现：

- YAML memory
- retrieval
- posterior prior blending
- update after selection

验收：

- 第二个 hammer-like scene 能检索 head/handle prior。
- 强视觉证据可以覆盖 memory。

## 18. 实验设计

### 18.1 Baselines

1. PhysGM global material
2. PartPhysAgent crop aggregation
3. MaterialAgent posterior MAP without video selection
4. MaterialAgent candidate video selection
5. Human selected candidate

### 18.2 测试物体

```text
hammer: metal head + wood/plastic handle
shoe: rubber sole + fabric/leather upper + laces
cake: foam/soft cake + frosting + plate
chair: rigid frame + cushion
toy car: plastic body + rubber wheels
cup: ceramic/glass body + possible liquid
```

### 18.3 指标

- material label accuracy
- solver success rate
- selected video preference rate
- physical role consistency
- deformation ordering correctness
- runtime
- candidate count
- VLM/human agreement

## 19. 风险和解决方案

### 风险 1: PhysGM variance 不校准

解决：

- 不只用 `E_var/nu_var`
- 加 crop disagreement
- 加 material table range
- 加 skill memory

### 风险 2: VLM 材料判断幻觉

解决：

- VLM 只做 prior
- 候选视频选择做闭环
- hard filters 防止坏候选入选

### 风险 3: AABB local params 污染邻近 part

解决：

- AABB 加 conservative padding
- 对 small part 限制半径
- Phase 2 改 per-particle material params

### 风险 4: solver 全局 constitutive class 限制

解决：

- MVP 明确只能 per-part `E/nu/density`
- metadata 保存 visual material
- Phase 2 修改 solver 支持 per-particle `material_id`

### 风险 5: 仿真成本高

解决：

- candidate budget 默认 5
- 只 sweep lowest-confidence part
- mock mode 单元测试
- hard filter early stop

## 20. Phase 2: 真正的 per-particle material class

当前原始 PhysGM:

```text
model.material 是全局 int
additional_material_params 不改 material class
```

Phase 2 需要：

1. 给每个 particle/gaussian 增加 `particle_material_id`
2. `apply_additional_params` 或新 kernel 同时写：

```text
E[p], nu[p], density[p], material_id[p]
```

3. stress/return mapping 改为按 particle material 分支，而不是全局 `model.material`
4. 支持：

```text
head -> metal/FCR
handle -> foam/plasticine/elastic
sole -> jelly
```

这才是真正意义上的 multi-material simulation。

## 21. 最终推荐路线

MVP 不要一上来改 solver。推荐顺序：

1. 先做 MaterialAgent 独立包，读取 `--segmentation-only` 的 PartPhysAgent 输出。
2. 把 PhysGM 的 `E_var/nu_var` 暴露出来，形成 per-part parameter posterior。
3. 生成 5 个候选 part-material config。
4. 跑 5 个候选视频。
5. 让用户或 VLM 选最好的，也提供 auto selector。
6. 把 winner 写成 `sim_config_materialagent_selected.json`。
7. 把 winner/failure 写进 skill memory。
8. 稳定后再做 per-particle material class solver 改造。

这条路线贡献明确、风险可控，也和 PhysGM 论文中“从物理参数分布采样多个候选视频并选择 winner”的思想一致，只是把单位从 whole object 推进到了 physical part。

