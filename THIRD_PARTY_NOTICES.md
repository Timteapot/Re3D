# 第三方资源说明

Re3D 的自定义编排和后处理脚本与以下第三方资源分开存放。使用或分发时应分别遵守各项目及模型权重的许可条件。

| 资源 | 固定版本/提交 | 本地位置 | 许可提示 |
|---|---|---|---|
| MapAnything | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | `vendor/map-anything` | 上游 `LICENSE` 为 Apache License 2.0 |
| facebook/map-anything 权重 | 本次缓存快照 | `models/mapanything` | 遵守模型发布页和上游仓库条款 |
| MVSAnywhere | `5bd49bba0992aa56060e1aa3bb40b2150b04726f` | `vendor/mvsanywhere` | 上游 `LICENSE` 标注 Niantic 版权所有及 All rights reserved；不要把本地副本视为可自由再分发资源 |
| MVSAnywhere hero 权重 | 本次实验 checkpoint | `models/mvsanywhere` | 遵守上游模型条款，仅按获准范围使用和分发 |
| DINOv2 | 本次 torch hub 缓存 | `models/torch/hub` | MVSAnywhere 的图像编码器依赖，遵守 DINOv2 上游许可 |
| OpenMVS | 2.4.0 Windows VC17 x64 | `vendor/openmvs-2.4.0-windows` | 当前二进制包未附带许可文件；再分发前应从 OpenMVS 上游补齐并核对许可文本 |

迁移仅在同一用户工作区内复制原项目已经使用的资源，没有改变模型、源码或二进制的权利归属。
