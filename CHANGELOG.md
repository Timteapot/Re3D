# Changelog

本文件按 Git 提交记录 `main` 的重要行为变化。日期使用仓库提交时间；实验产物、`.cache` 和本机配置不列为受版本控制的发布文件。

## Unreleased

## re3d-pipeline-v1.1.2 — 2026-10-10 — MVSAnywhere 离线加载修复

- 将 MVSAnywhere 的 DINOv2 Torch Hub 引用固定为 `facebookresearch/dinov2:main`，使已准备的 `facebookresearch_dinov2_main` 缓存无需先访问 GitHub 即可确定性加载。
- 修正 MVSAnywhere 手动检查点回退顺序：先删除分辨率相关缓存张量，再以非严格模式加载权重，避免网络异常掩盖为尺寸不匹配错误。
- 保持 A-v4 的 Tesla T4 批量参数、三分支算法参数、模型权重和输出契约不变。

## re3d-pipeline-v1.1.1 — 2026-10-10 — T4 兼容基线

- 将 A-v4 MapAnything 的默认批量大小从 12 调整为 4，避免 15 GiB NVIDIA T4 在多视角注意力阶段发生 CUDA OOM。
- 同步 `run_mapanything_batched.py` 的独立入口默认值；分支、模型、深度校正、跨视角过滤和 OpenMVS 参数保持不变。
- 该值已经在目标 Windows Server、Tesla T4 和 11 张 2832×2128 图片上完成三分支真实重建验收。

### 文档与复现信息

- 更新 `README.md`，明确当前可运行管线、入口、依赖和 OpenMVS 混合调用。
- 新增 `docs/architecture/pipeline.md`，记录 A/B/C 调用链、共享阶段和文件位置。
- 新增 `docs/build/openmvs.md`，记录修复版 TextureMesh 的来源、VC18 Release 依赖、恢复、构建、配置、校验与风险。
- 将包含 `eeedab7` 修复的 `33d9484` VC18 完整 Release 从 `.cache` 迁移到 `vendor/openmvs-2.4.0-33d9484-windows/`，使 TextureMesh 及同次构建 DLL 可随 Git 恢复。
- 更新 `configs/paths.example.json` 和本机配置，指向新的 vendor 相对路径。
- `scripts/run_pipeline.py` 的 TextureMesh 无覆盖默认值改为修复版 vendor；环境变量和本机配置覆盖规则保持不变，几何程序仍使用原 VC17 vendor。
- 新增默认路径测试；不改变重建算法、分支调用链或数值参数。

## ae4f71f — 2026-09-22 — Add safe override for fixed OpenMVS texturing

- `scripts/run_pipeline.py` 新增 `RE3D_TEXTUREMESH_EXE` 与 `texturemesh_executable` 解析，纹理阶段可独立选择修复版二进制。
- 接缝平衡启用时拒绝仓库内已知不安全的旧 OpenMVS 2.4.0 TextureMesh。
- 新增对应测试；`.cache/` 纳入忽略规则。
- 更新 README 与路径示例，说明上游 `eeedab7` 修复要求。

## 03e865a — 2026-09-22 — Use conservative texture filtering without seam leveling

- `configs/pipeline.json` 默认关闭全局与局部接缝平衡。
- 保留锐度权重 `0.25` 与离群阈值 `0.06`，降低黑纹理回归风险。

## f5a9a50 — 2026-09-20 — Add deferred COLMAP registration recovery

- 新增 `scripts/sfm_quality.py` 和 `tests/test_sfm_quality.py`。
- 共享 COLMAP 前端可识别弱 2D–3D 约束视角，将其移出核心建图并在核心相机固定后以更严格阈值重注册。
- 扩展 `configs/pipeline.json`、编排器和测试，记录注册质量与恢复结果。

## 4885d76 — 2026-09-20 — Use alpha masks for OpenMVS PatchMatch

- C 路线的 OpenMVS PatchMatch 显式接收 alpha 前景掩码和忽略标签，避免把透明背景当作稠密几何。
- 调整 OpenMVS 纹理输入准备、配置、编排和接口测试。

## c014826 — 2026-09-20 — Merge remote main into optimized pipeline

- 合并远端 `main` 与优化管线历史；没有独立的运行行为增量。

## 8cd1614 — 2026-09-20 — Freeze optimized OpenMVS handoff pipeline

- 固定 A-v4、B-v2 与 C 的统一 OpenMVS 后端管线。
- 新增 type-5 DMAP 读写 `scripts/openmvs_dmap.py`、接口验证 `validate_openmvs_handoff.py`、canonical/staging 隔离 `stage_openmvs_handoff.py` 和网格质量评估 `evaluate_mesh_quality.py`。
- A/B 深度在稀疏尺度校正和跨视角过滤后转换为兼容 DMAP，再交给 OpenMVS 融合与网格重建。
- 新增 `tests/test_openmvs_handoff.py`，加强相机、尺寸、payload、数值和不可变性约束。

## fdd0dfb — 2026-09-17 — Update README.md

- 补充项目说明性备注，不改变管线行为。

## 239dc97 — 2026-09-17 — Update project documentation

- 更新初始 README，整理项目说明，不改变管线行为。

## c5533bd — 2026-09-17 — Initial project import

- 导入 Re3D 初始代码、配置、第三方依赖、模型元数据和 Windows 入口脚本，形成后续优化基线。
