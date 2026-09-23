# Re3D 管线架构

本文描述 `main` 当前实现。运行参数以 [`configs/pipeline.json`](../../configs/pipeline.json) 为准，入口与阶段编排以 [`scripts/run_pipeline.py`](../../scripts/run_pipeline.py) 为准。

## 总体调用链

```text
run.ps1
  -> scripts/run_pipeline.py
     -> 输入校验、复制、alpha 掩码和哈希清单
     -> pycolmap CPU SIFT + 穷举匹配 + 增量建图
     -> 弱约束视角延迟重注册
     -> 相机导出 + OpenMVS 输入准备
     -> C 几何前置步骤：InterfaceCOLMAP -> DensifyPointCloud
        ├─ C：ReconstructMesh -> TextureMesh -> 归一化/验证
        ├─ A：MapAnything -> 稀疏尺度校正 -> 跨视角 v4
        │     -> DMAP 导出/校验/暂存 -> DensifyPointCloud 融合
        │     -> ReconstructMesh -> TextureMesh -> 归一化/验证
        └─ B：MVSAnywhere -> 稀疏尺度校正 -> 跨视角 v2
              -> DMAP 导出/校验/暂存 -> DensifyPointCloud 融合
              -> ReconstructMesh -> TextureMesh -> 归一化/验证
```

A/B 依赖 C 稠密化产生、且与相机一致的 DMAP 模板。即使 `-Branches a` 或 `-Branches b`，`InterfaceCOLMAP` 和 C 的 PatchMatch 稠密化仍会执行；只有 C 的网格重建与纹理化会被跳过。

## 共享步骤

| 阶段 | 实现 | 主要输入 | 主要输出 |
|---|---|---|---|
| 数据准备 | `scripts/prepare_dataset.py` | PNG/JPG/JPEG | `work/<scene>/input/images`、`masks`、`input-manifest.json` |
| 相机估计 | `scripts/run_colmap_shared.py` | 图像、alpha 掩码 | `work/<scene>/shared/colmap`、`reconstruction_metrics.json` |
| 弱视角恢复 | `scripts/sfm_quality.py` 与 `run_colmap_shared.py` | 初次稀疏模型 | 固定核心模型后的延迟重注册结果 |
| 相机导出 | `scripts/extract_colmap_cameras.py` | COLMAP refined model | `work/<scene>/shared/cameras_refined.npz` |
| OpenMVS 输入 | `scripts/prepare_openmvs_texture_input.py` | 图像、掩码、稀疏模型 | `work/<scene>/shared/openmvs_input` |
| C 几何前置 | `InterfaceCOLMAP.exe`、`DensifyPointCloud.exe` | OpenMVS 输入 | `work/<scene>/branches/c_openmvs/scene.mvs`、`scene_dense.mvs` 与 DMAP |
| 网格 | `ReconstructMesh.exe` | 各分支的 `scene_dense.mvs` | `scene_mesh.mvs`、`scene_mesh.ply` |
| 纹理 | 修复版 `TextureMesh.exe` | `scene.mvs`、`scene_mesh.ply`、原图 | OBJ、MTL、纹理图 |
| 归一化与验证 | `normalize_openmvs_output.py`、`validate_outputs.py` | OpenMVS 纹理结果 | `outputs/<scene>/<branch>`、`validation.json` |

输入图像带 alpha 时，准备阶段生成前景掩码；C 的 `DensifyPointCloud` 通过 `--mask-path` 和 `--ignore-mask-label 0` 排除透明背景。RGB/JPEG 没有透明信息，会得到全前景掩码。

## 运行目录边界

默认情况下，单个场景继续使用 `work/<scene>`、`outputs/<scene>` 和 `logs/<scene>`，保持现有脚本与历史运行兼容。编排器可以通过 `--work-dir`、`--output-dir`、`--log-dir` 为单次运行指定精确目录，或者使用 `RE3D_WORK_DIR`、`RE3D_OUTPUT_DIR`、`RE3D_LOG_DIR`。命令行参数优先于环境变量。

这些参数只改变运行数据位置，不改变管线参数、模型、分支或产物命名。显式目录不会再追加 `scene`，因此 Web Worker 可以把一个 UUID 任务严格映射为：

```text
jobs/<job_uuid>/
├── runtime/work/    # --work-dir
├── runtime/logs/    # --log-dir
└── output/          # --output-dir
```

Re3D 仍假设调用方可信，不负责鉴权或多租户路径隔离。公开服务必须在调用前完成 UUID、规范化绝对路径和任务根目录边界检查。

## A 路线：MapAnything A-v4

1. `run_mapanything_batched.py` 批量预测深度，使用 `--skip-tsdf`。
2. `calibrate_depths_to_colmap_sparse.py` 用逐图 COLMAP 稀疏点恢复尺度。
3. `optimize_multiview_depths.py` 执行 v4 跨视角一致性过滤。
4. `export_model_depths_to_openmvs.py` 以 C 的 DMAP 为模板写出 OpenMVS type-5 depth+confidence DMAP。
5. `validate_openmvs_handoff.py` 检查尺寸、相机、view ID、payload、数值范围和反投影稳定性。
6. `stage_openmvs_handoff.py` 把 immutable canonical 输入复制到空 staging，并在融合后复核哈希。
7. 原 vendor `DensifyPointCloud.exe` 以 `fusion-filter=2` 融合，再由原 vendor `ReconstructMesh.exe` 建网格。
8. 修复版 `TextureMesh.exe` 生成纹理网格。

主要目录：

```text
work/<scene>/branches/
├── a_mapanything/
│   ├── raw/
│   ├── depth-sparse-calibrated/
│   └── depth-v4/
├── a_openmvs_input/     # canonical DMAP
└── a_openmvs/           # staging 与 OpenMVS 可变产物
```

## B 路线：MVSAnywhere B-v2

B 与 A 共享尺度校正、DMAP Adapter、Validator、canonical/staging 隔离、OpenMVS 融合、网格和纹理后端。区别是：

1. `run_mvsanywhere.py` 使用 MVSAnywhere hero、7 个源视角和 384×384 推理。
2. `optimize_multiview_depths.py` 使用 B-v2 一致性参数。
3. 原 vendor `DensifyPointCloud.exe` 融合 B 的模型深度；修复版 `TextureMesh.exe` 仅负责最终纹理化。

主要目录与 A 同构，分别为 `b_mvsanywhere/`、`b_openmvs_input/` 和 `b_openmvs/`，均位于 `work/<scene>/branches/`。

## C 路线：传统 OpenMVS PatchMatch

1. 原 vendor `InterfaceCOLMAP.exe` 把共享 COLMAP 模型转换为 `scene.mvs`。
2. 原 vendor `DensifyPointCloud.exe` 在 alpha mask 约束下运行 PatchMatch，并输出 C 的稠密点云和 DMAP 模板。
3. 原 vendor `ReconstructMesh.exe` 重建网格。
4. 修复版 `TextureMesh.exe` 纹理化。

C 不使用 A/B 的学习深度、尺度校正或 DMAP Adapter。

## OpenMVS 混合调用边界

当前 `main` 不是整套切换到新 OpenMVS，而是按进程混合调用：

| 程序 | 实际位置 | 选择方式 |
|---|---|---|
| `InterfaceCOLMAP.exe` | `vendor/openmvs-2.4.0-windows/vc17/x64/Release/` | 固定，由 Git 管理 |
| `DensifyPointCloud.exe` | 同上 | 固定，由 Git 管理 |
| `ReconstructMesh.exe` | 同上 | 固定，由 Git 管理 |
| `TextureMesh.exe` | `vendor/openmvs-2.4.0-33d9484-windows/vc18/x64/Release/` | `RE3D_TEXTUREMESH_EXE` 优先，其次 `configs/paths.local.json`，最后使用该修复版 vendor 默认值 |

修复版 `TextureMesh.exe` 必须从自己的完整 `vc18/x64/Release` 目录加载同次构建的 `MVS.dll`、`Common.dll`、`IO.dll`、`Math.dll`、OpenCV、Boost 及其他传递依赖。不要只把 exe 复制到 VC17 目录，也不要把 VC18 DLL 复制进 VC17 目录。具体恢复方法见 [`docs/build/openmvs.md`](../build/openmvs.md)。

这种混合的边界是磁盘上的 OpenMVS `.mvs`、PLY、图像和纹理文件，不是在同一进程内混载两组 DLL。当前固定组合已经过项目实验验证；更换 OpenMVS 提交后仍需重新验证序列化兼容性和纹理结果。

## 主要代码与配置位置

| 路径 | 作用 |
|---|---|
| `run.ps1` | PowerShell 统一入口 |
| `doctor.ps1` | 环境、模型和可执行文件检查 |
| `scripts/run_pipeline.py` | 阶段编排、缓存标记、OpenMVS 程序选择 |
| `configs/pipeline.json` | A/B/C、网格和纹理参数 |
| `configs/paths.example.json` | 受 Git 管理的可移植路径模板 |
| `configs/paths.local.json` | 本机路径；被 `.gitignore` 忽略 |
| `vendor/openmvs-2.4.0-windows/vc17/x64/Release` | 原 OpenMVS 几何工具与运行库 |
| `vendor/openmvs-2.4.0-33d9484-windows/vc18/x64/Release` | 修复版 TextureMesh 完整运行目录；随仓库受 Git 管理 |
| `work/<scene>` | 中间产物与阶段标记；被 Git 忽略 |
| `outputs/<scene>` | 最终模型与验证报告；被 Git 忽略 |
| `logs/<scene>` | 各阶段日志；被 Git 忽略 |

## 入口命令

```powershell
# 环境检查
.\doctor.ps1

# 仅打印将执行的命令
.\run.ps1 -Scene example -Images D:\photos\example -Branches all -DryRun

# 全部分支或单分支
.\run.ps1 -Scene example -Images D:\photos\example -Branches all
.\run.ps1 -Scene example_c -Images D:\photos\example -Branches c
```

阶段完成标记不会因配置或输入变化自动失效。对比实验应使用新的 `Scene` 名称，避免复用旧的 `work/<scene>`。
