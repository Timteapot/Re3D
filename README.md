# Re3D

Re3D 是从 `experiments/feedforward-mvs` 整理出的可复用三分支重建管线。迁移保留当前选定方案：

- A：MapAnything + 稀疏尺度校正 + 跨视角 v4 + OpenMVS；
- B：MVSAnywhere + 稀疏尺度校正 + 跨视角 v2 + OpenMVS；
- C：COLMAP + OpenMVS PatchMatch 基准路线。

历史 LEGO/poster 输入、实验输出、日志和中间深度没有复制。原实验目录保持不变。

## 目录

```text
Re3D/
├── README.md
├── run.ps1                         # 统一运行入口
├── doctor.ps1                      # 环境与资源检查
├── configs/
│   ├── pipeline.json               # 当前 A-v4/B-v2/C 参数
│   ├── paths.local.json            # 本机 Python 环境路径
│   ├── paths.example.json
│   └── reference-*.json            # 原实验冻结配置，仅供核对
├── scripts/
│   ├── run_pipeline.py             # 总体编排
│   ├── prepare_dataset.py           # 输入复制、掩码和哈希清单
│   ├── run_colmap_shared.py
│   ├── run_mapanything_batched.py
│   ├── run_mvsanywhere.py
│   ├── calibrate_depths_to_colmap_sparse.py
│   ├── optimize_multiview_depths.py
│   ├── export_model_depths_to_openmvs.py
│   ├── prepare_openmvs_texture_input.py
│   ├── normalize_openmvs_output.py
│   └── validate_outputs.py
├── vendor/
│   ├── map-anything/               # 固定源码副本
│   ├── mvsanywhere/                # 固定源码副本
│   └── openmvs-2.4.0-windows/      # 当前验证过的 Windows 二进制
├── models/
│   ├── mapanything/                # config.json + model.safetensors
│   ├── mvsanywhere/                # mvsanywhere_hero.ckpt
│   ├── torch/hub/                  # DINOv2 源码缓存与 vitb14 权重
│   └── checksums.json
├── environments/                   # 环境快照，不包含不可移植的虚拟环境
├── data/scenes/                    # 可选的用户输入放置区
├── work/<scene>/                   # 相机、深度、DMAP、点云、网格等中间数据
├── outputs/<scene>/                # 最终 OBJ/MTL/JPG/GLB
└── logs/<scene>/                   # 分阶段日志
```

## 当前机器直接运行

先检查依赖：

```powershell
cd D:\3Dreconstruction\Re3D
.\doctor.ps1
```

运行三条分支：

```powershell
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches all
```

只运行指定分支：

```powershell
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches a
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches b
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches c
```

A/B 依赖 C 生成相机一致的 DMAP 模板。只选择 A 或 B 时，编排器会自动执行 C 的相机转换和 PatchMatch 深度阶段，但不会生成 C 的最终纹理模型。

预览将执行的命令而不运行：

```powershell
.\run.ps1 -Scene my_object -Images D:\photos\my_object -Branches all -DryRun
```

编排器按产物标记自动续跑。阶段产物已存在时会跳过，不会覆盖已完成结果。需要从头实验时应使用新的 `-Scene` 名称。

## 输入要求

- 支持 PNG、JPG 和 JPEG；
- 至少 3 张图片，实际重建建议远多于最低值；
- 当前版本要求所有图片尺寸一致；
- 当前 COLMAP 前端按单相机 PINHOLE 模型处理，适合固定焦距拍摄；
- RGBA 输入会使用 alpha 生成前景掩码；普通 RGB/JPEG 会生成全前景掩码；
- 普通照片若背景复杂，建议在运行前提供已经抠图的 PNG，或先增加独立的分割步骤；
- 图片应保持静态场景、连续视角和充分重叠。

## 运行顺序

```text
输入整理
  → COLMAP SIFT/匹配/增量建图
  → 相机导出与 OpenMVS RGB/alpha 输入
  → C PatchMatch（同时为 A/B 生成 DMAP 模板）
  → A MapAnything 或 B MVSAnywhere 深度
  → COLMAP 稀疏点逐图尺度校正
  → A-v4 / B-v2 跨视角过滤
  → 深度写入 DMAP
  → OpenMVS 融合与 ReconstructMesh
  → TextureMesh
  → OBJ/MTL/JPG/GLB 归一化与加载验证
```

A/B 推理脚本在迁移版中使用 `--skip-tsdf`，不再生成已经被 OpenMVS 替代的 Open3D TSDF 临时网格，因此减少内存、运行时间和无用中间文件。

## 最终输出

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

`work/<scene>` 可以在确认最终产物后自行归档或删除。编排器不会自动删除中间数据。

## 环境

现有 Conda 环境没有直接复制。两个环境合计约 12.7 GB，并含有与原安装位置绑定的二进制和绝对路径，简单移动不能保证可用。

当前机器的可执行路径记录在 `configs/paths.local.json`。迁移到其他机器后，可以修改该文件，或设置：

```powershell
$env:RE3D_MAP_PYTHON = 'D:\envs\mapanything\python.exe'
$env:RE3D_MVS_PYTHON = 'D:\envs\mvsanywhere\python.exe'
```

`environments/` 保存了本次成功环境的 Conda 和 pip 快照。环境重建说明见 `environments/README.md`。
推荐的新环境位置为 `environments/runtime/`；它与固定依赖快照分开并已加入 `.gitignore`。

## 结果解释

Re3D 保留的是已验证管线和参数，不包含“对任意数据保证质量”的假设。COLMAP 注册失败、透明/反光材质、动态场景、弱纹理和视角覆盖不足仍可能导致失败。A/B/C 的最终质量应结合相机注册率、跨视角深度保留率、网格碎片数和可视检查判断。

第三方源码、模型和二进制的归属与许可说明见 `THIRD_PARTY_NOTICES.md`。
