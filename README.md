# Re3D

## 当前技术管线

Re3D 是一套面向多视图照片的三分支 3D 重建管线。三个分支共享 COLMAP 相机估计和 OpenMVS 网格/纹理后端，用于比较学习型深度与传统 PatchMatch 深度在同一相机、融合和纹理条件下的结果。

| 分支 | 深度来源 | 深度处理 | 几何与纹理后端 |
|---|---|---|---|
| **A-v4** | MapAnything 批量推理 | COLMAP 稀疏点逐图尺度校正；跨视角 v4 一致性过滤 | OpenMVS 深度融合、网格重建和纹理化 |
| **B-v2** | MVSAnywhere hero，7 个源视角，384×384 | COLMAP 稀疏点逐图尺度校正；跨视角 v2 一致性过滤 | OpenMVS 深度融合、网格重建和纹理化 |
| **C** | COLMAP 相机 + OpenMVS PatchMatch | alpha 前景掩码约束的 OpenMVS 原生稠密化 | OpenMVS 网格重建和纹理化 |

共享前端与完整执行流：

```text
PNG/JPG/JPEG
  → 输入校验、复制、alpha 掩码与哈希清单
  → pycolmap CPU SIFT 特征、穷举匹配、固定随机种子的增量建图
  → 逐图稀疏观测质量门控；低置信度视角移出核心模型后延迟重注册
  → 相机导出与 OpenMVS 场景准备
  → C 分支使用显式 alpha mask 的 PatchMatch 深度（同时生成 A/B 所需的 DMAP 模板）
  ├─→ C：融合 → ReconstructMesh → TextureMesh
  ├─→ A：MapAnything → 稀疏尺度校正 → 跨视角 v4 → DMAP Adapter
  │       → Validator → immutable canonical → 单次 staging
  │       → OpenMVS filter=2 融合 → ReconstructMesh → TextureMesh
  └─→ B：MVSAnywhere → 稀疏尺度校正 → 跨视角 v2 → DMAP Adapter
          → Validator → immutable canonical → 单次 staging
          → OpenMVS filter=2 融合 → ReconstructMesh → TextureMesh
  → OBJ/MTL/JPG/GLB 归一化与加载验证
```

A/B 均依赖 C 分支生成与相机一致的 DMAP 模板。因此，即使只选择 A 或 B，C 的相机转换和 PatchMatch 稠密化仍会执行，但不会生成 C 的最终纹理网格。A/B 推理使用 `--skip-tsdf`，不生成被 OpenMVS 后端替代的 Open3D TSDF 临时网格。Adapter 输出 depth+confidence type-5 DMAP；Validator 在 OpenMVS 启动前检查尺寸、相机、view ID、payload 长度、数值范围和反投影稳定性。融合只操作 staging 副本，并在运行后复核 canonical SHA-256。

当前关键参数由 [`configs/pipeline.json`](configs/pipeline.json) 统一管理：

- A-v4：批量大小 12，7 个相邻视角，内部最少支持数 1、边缘最少支持数 2，相对深度阈值 0.05，重投影阈值 2 px；
- B-v2：7 个源视角，1 次 refinement，7 个相邻视角，最少支持数 2，相对深度阈值 0.04，重投影阈值 2 px；
- A/B：每张图至少使用 30 个 COLMAP 稀疏点完成尺度校正；
- COLMAP：低于场景三角化观测中位数 15% 且特征三角化率低于 5% 的视角进入延迟队列；固定核心相机后，以更严格的绝对位姿阈值重试，仍不可靠的视角不进入后续分支；
- C 与 A/B 共用稠密分辨率契约：`resolution-level=1`、`min-resolution=640`、`max-resolution=1024`；
- C 的 PatchMatch 使用 6 个视角，并通过 `mask-path`、`ignore-mask-label=0` 排除透明背景；A/B 正式融合使用 `number-views-fuse=2`、`fusion-filter=2`；
- 网格：去除孤立成分 4、补洞 30、平滑 2；纹理最大尺寸 8192；全局/局部接缝平衡默认关闭，保留锐度权重 `0.25` 和离群阈值 `0.06`。

## 快速开始

*我觉得还是别开始了，光配置配半天，还是脆弱的windows环境（不必移除，这只是一句吐槽）*

### 1. 配置本机环境

首次使用时，从示例创建本机配置并填写两个 Python 解释器路径：

```powershell
cd D:\3Dreconstruction\Re3D
Copy-Item .\configs\paths.example.json .\configs\paths.local.json
```

也可以用环境变量临时覆盖配置：

```powershell
$env:RE3D_MAP_PYTHON = 'D:\envs\mapanything\python.exe'
$env:RE3D_MVS_PYTHON = 'D:\envs\mvsanywhere\python.exe'
```

如需实验性重新启用 OpenMVS 全局或局部接缝平衡，必须通过
`configs/paths.local.json` 的 `texturemesh_executable` 或环境变量
`RE3D_TEXTUREMESH_EXE` 指向包含上游修复 `eeedab7` 的 `TextureMesh.exe`。
管线会拒绝让仓库自带的 OpenMVS 2.4.0 构建执行接缝平衡，以避免已知的大面积黑纹理回归。

大型模型权重不纳入 Git，运行前需确认以下文件存在：

```text
models/mapanything/model.safetensors
models/mvsanywhere/mvsanywhere_hero.ckpt
models/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
```

### 2. 检查依赖

```powershell
.\doctor.ps1
```

检查项包括两个 Python 环境、核心 Python 包、模型权重、第三方源码和 OpenMVS 可执行文件。

### 3. 预览或运行

先预览命令而不执行重建：

```powershell
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches all -DryRun
```

运行全部分支：

```powershell
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches all
```

运行指定分支：

```powershell
.\run.ps1 -Scene my_object_a -Images D:\photos\my_object -Branches a
.\run.ps1 -Scene my_object_b -Images D:\photos\my_object -Branches b
.\run.ps1 -Scene my_object_c -Images D:\photos\my_object -Branches c
.\run.ps1 -Scene my_object_ab -Images D:\photos\my_object -Branches a,b
```

`Scene` 只允许字母、数字、点、下划线和连字符。新场景必须提供 `-Images`；已有场景续跑时可以省略。

## 输入要求

- 支持 `.png`、`.jpg` 和 `.jpeg`；
- 至少 3 张图片，稳定重建通常需要更多连续视角；
- 同一场景的图片尺寸必须一致；
- COLMAP 前端使用单相机 `PINHOLE` 模型，适合固定焦距拍摄；
- RGBA 图片根据 alpha 通道生成前景掩码，阈值默认为 128；
- RGB/JPEG 会生成全前景掩码，复杂背景建议预先分割为带 alpha 的 PNG；
- 拍摄对象应保持静止，并具有充分视角重叠、纹理和清晰度；
- 透明、镜面、重复纹理、弱纹理和运动物体会显著降低相机估计或深度融合质量。

## 目录结构

```text
Re3D/
├── run.ps1                         # PowerShell 统一入口
├── doctor.ps1                      # 环境和资源检查
├── configs/
│   ├── pipeline.json               # 当前管线参数
│   ├── paths.example.json          # 本机路径配置示例
│   └── paths.local.json            # 本机路径配置，不纳入 Git
├── scripts/
│   ├── run_pipeline.py             # 阶段编排、缓存和日志
│   ├── prepare_dataset.py          # 输入、掩码与哈希清单
│   ├── run_colmap_shared.py        # 共享 COLMAP 前端
│   ├── run_mapanything_batched.py  # A 分支深度推理
│   ├── run_mvsanywhere.py          # B 分支深度推理
│   ├── calibrate_depths_to_colmap_sparse.py
│   ├── optimize_multiview_depths.py
│   ├── export_model_depths_to_openmvs.py
│   ├── openmvs_dmap.py             # DMAP 读写与格式定义
│   ├── validate_openmvs_handoff.py # 融合前只读接口校验
│   ├── stage_openmvs_handoff.py    # canonical/staging 隔离和哈希复核
│   ├── evaluate_mesh_quality.py    # 网格连通性与边界指标
│   ├── normalize_openmvs_output.py
│   └── validate_outputs.py
├── vendor/                         # 固定的第三方源码和 OpenMVS 二进制
├── models/                         # 模型配置、权重和 DINOv2 缓存
├── environments/                   # 可复现环境的依赖快照
├── data/scenes/                    # 可选的本地输入区
├── work/<scene>/                   # 相机、深度、DMAP、点云与网格中间文件
├── outputs/<scene>/                # 最终模型和验证报告
└── logs/<scene>/                   # 每个阶段的完整日志
```

`work/`、`outputs/`、`logs/`、本机环境、输入场景和模型权重默认不纳入 Git。

## 续跑与实验隔离

编排器以阶段产物作为完成标记。标记存在时会跳过对应阶段，不会自动覆盖已经完成的结果；失败阶段可在修复问题后使用相同命令继续执行。A/B 的 canonical DMAP 与 OpenMVS 可变运行目录分离；staging 必须为空，且会在启动融合前验证 scene 中的相对图像路径。

修改模型、输入或 [`configs/pipeline.json`](configs/pipeline.json) 后，已有完成标记不会自动失效。进行参数对比或从头重建时，应使用新的 `Scene` 名称，避免新旧产物混用。确认最终结果后，可以手动归档或删除 `work/<scene>`；程序不会自动清理中间数据，后续需要设置独立的测试报告文件夹，按实验名称和时间，多文件层次输出。

## 输出与评估

```text
outputs/<scene>/
├── A-v4/
│   ├── mesh.obj
│   ├── mesh.mtl
│   ├── texture.jpg
│   ├── mesh.glb
│   └── artifact_manifest.json
├── B-v2/
├── C/
└── validation.json
```

主要诊断文件：

| 阶段 | 文件 | 用途 |
|---|---|---|
| 相机重建 | `work/<scene>/shared/colmap/reconstruction_metrics.json` | 检查注册图像数和稀疏重建质量 |
| A/B 尺度校正 | `sparse_calibration_manifest.json` | 检查每张图的稀疏点数量和尺度估计 |
| A/B 跨视角过滤 | `consistency_manifest.json` | 检查深度保留率和一致性过滤结果 |
| A/B DMAP 导出 | `dmap-export.json` | 记录 schema、深度/置信度语义、来源哈希、相机模板和分辨率契约 |
| A/B 接口校验 | `<branch>_handoff_validation.json` | 在融合前检查 DMAP 格式、相机、view ID、数值和重投影稳定性 |
| A/B 输入隔离 | `staging-manifest.json`、`<branch>_handoff_immutability.json` | 记录复制哈希并确认 canonical 未被 OpenMVS 修改 |
| 最终验证 | `outputs/<scene>/validation.json` | 检查 OBJ、MTL、纹理和 GLB 是否可加载 |
| 阶段日志 | `logs/<scene>/*.log` | 定位外部程序或脚本失败原因 |

`validation.json` 通过只表示产物结构完整且能够加载，不代表几何质量达到预期。A/B/C 应结合相机注册率、尺度校正覆盖、跨视角深度保留率、网格连通性和可视结果共同评价。

## 优化与调参建议

1. **先保证输入质量。** 补充连续视角、减少模糊并保持固定焦距，通常比放宽后处理阈值更有效。
2. **先检查 COLMAP。** 三个分支共享相机结果；注册率不足时，应优先改善拍摄、掩码或匹配，而不是调整 A/B 深度参数。
3. **再检查尺度校正。** 若多数图片达不到 `minimum_sparse_points`，应先提高稀疏重建覆盖或调整逐图尺度策略。
4. **最后调跨视角过滤。** 输出缺失过多时可逐步降低支持数或放宽深度/重投影阈值；漂浮噪声较多时反向收紧。每次只改变一组参数并使用新场景名对比。
5. **统一后端比较。** A/B/C 共用网格和纹理配置。比较深度方案时应保持 OpenMVS 参数不变；优化最终成品时再单独调整 `mesh` 和 `texture`。

## 环境

当前验证组合：

- MapAnything：Python 3.12、PyTorch 2.7.1+cu128；
- MVSAnywhere：Python 3.10、PyTorch 2.1.2+cu118；
- OpenMVS：2.4.0 Windows VC17 x64；
- MVSAnywhere 图像编码器依赖项目内的 DINOv2 源码缓存和 ViT-B/14 权重。

`environments/` 保存 Conda 和 pip 依赖快照，具体重建步骤见 [`environments/README.md`](environments/README.md)。CUDA、显卡驱动和 PyTorch wheel 必须相互兼容。

## 第三方许可

第三方源码、模型权重和二进制文件分别受各自许可约束。使用或分发前请阅读 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 以及各上游项目附带的许可文件。虽然感觉就此综设项目而言估计没啥必要，但是版权还是很重要的就留着吧。
