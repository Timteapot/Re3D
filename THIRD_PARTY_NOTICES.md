# 第三方资源、许可与署名说明

Re3D 是用于学习、研究和实践真实 3D 重建项目流程的非商业项目。本项目及其部署实例不收费、不盈利，也不用于代表商业实体提供服务。该用途声明不会替代或放宽任何第三方许可证；每项第三方代码、模型、权重和二进制文件仍分别受其原始条款约束。

本文记录当前仓库已知的直接第三方资源、来源、固定版本、许可边界和部署展示要求。它不是法律意见。若项目用途、部署对象、权重来源或分发方式发生变化，应重新审查全部条款。

## 使用与再分发是两件不同的事

- 在本机或受控学习环境中运行第三方资源，不等于可以把其源码、模型权重或二进制文件重新打包发布。
- 网页仅提供非商业学习演示时，仍应展示第三方署名、许可证链接、无担保说明及模型来源。
- 公开 Git 仓库、容器镜像、安装包和可下载模型均属于需要单独检查的分发场景。
- 模型生成的 OBJ、GLB、纹理和其他输出不会因此自动获得第三方软件的所有权；但输入图片、训练数据、模型权重及具体许可证可能对输出用途另有约束。

## 核心算法、模型与二进制

| 资源 | 当前固定版本或文件 | 本地位置 | 许可与使用边界 | 上游来源 |
|---|---|---|---|---|
| MapAnything 源码 | `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9` | `vendor/map-anything` | 源码为 Apache License 2.0。源码许可证不自动覆盖模型权重。 | [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) |
| MapAnything 权重 | `models/mapanything/model.safetensors`；SHA-256 `981F060C64664DFF3272B5F5A823D350ABE71A2F144444DB4CFC325F3ED5A3A0` | `models/mapanything` | 当前仓库没有保存足以唯一证明该文件对应哪一个 Hugging Face 模型变体的来源元数据。官方默认 `facebook/map-anything` 为 CC-BY-NC 4.0，`facebook/map-anything-apache` 为 Apache 2.0。在来源补录前，本项目按更严格的 CC-BY-NC 4.0、仅非商业用途处理该权重。 | [官方模型许可说明](https://github.com/facebookresearch/map-anything#models) |
| MVSAnywhere 源码 | `5bd49bba0992aa56060e1aa3bb40b2150b04726f` | `vendor/mvsanywhere` | Niantic 自定义 MVSAnywhere/DoubleTake License，仅允许非商业用途，并包含署名和再分发限制。不得把本地副本视为普通开源依赖或自由再分发资源。 | [nianticlabs/mvsanywhere](https://github.com/nianticlabs/mvsanywhere)；[LICENSE](https://github.com/nianticlabs/mvsanywhere/blob/main/LICENSE) |
| MVSAnywhere hero 权重 | `models/mvsanywhere/mvsanywhere_hero.ckpt`；SHA-256 `00878BAB903B1D384CD6BB9EA092E247A9036D429CB8BCC3D6DCBB81F67E556D` | `models/mvsanywhere` | 仅用于当前非商业学习和研究实践；不得随公开安装包、镜像或模型下载重新发布，除非上游条款明确允许或已取得书面许可。 | [MVSAnywhere 预训练模型说明](https://github.com/nianticlabs/mvsanywhere#pretrained-models) |
| DINOv2 源码缓存 | 当前本地 torch hub 快照 | `models/torch/hub/facebookresearch_dinov2_main` | 本地源码附带 Apache License 2.0；其第三方子目录仍按各自许可证执行。 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| DINOv2 ViT-B/14 权重 | `models/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`；SHA-256 `0B8B82F85DE91B424ADED121C7E1DCC2B7BC6D0ADEEA651BF73A13307FAD8C73` | `models/torch/hub/checkpoints` | 作为 MVSAnywhere 图像编码器依赖使用；保留上游来源和权重条款，不随项目公开分发。 | [DINOv2 模型说明](https://github.com/facebookresearch/dinov2#pretrained-models) |
| OpenMVS 几何工具 | OpenMVS 2.4.0 Windows VC17 x64 | `vendor/openmvs-2.4.0-windows` | OpenMVS 为 GNU AGPL v3。当前目录包含二进制和运行库，但没有随包附带完整许可证文件；分发前必须补充许可证文本、对应源码获取方式和必要的源代码提供信息。 | [cdcseacave/openMVS](https://github.com/cdcseacave/openMVS)；[LICENSE](https://github.com/cdcseacave/openMVS/blob/develop/LICENSE) |
| OpenMVS 纹理工具 | `33d9484` Windows VC18 x64，包含上游修复 `eeedab7` | `vendor/openmvs-2.4.0-33d9484-windows` | 同样受 GNU AGPL v3 约束。Re3D 通过独立进程和磁盘文件调用它；若公开提供网络服务或分发修改版，应再次核对 AGPL 的对应源码和网络交互义务。 | [固定修复提交](https://github.com/cdcseacave/openMVS/commit/eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d) |
| COLMAP / pycolmap | 版本由 `environments/` 依赖快照固定 | Python 运行环境 | COLMAP/pycolmap 使用 BSD 3-Clause License；应保留版权和许可声明。 | [colmap/colmap](https://github.com/colmap/colmap) |

模型文件未纳入 Git；文件身份由 [`models/checksums.json`](models/checksums.json) 记录。校验值只能证明文件未变化，不能单独证明下载来源或授权范围。

## Python 直接运行时依赖

Re3D 自定义脚本直接使用下列主要 Python 包。具体版本以 [`environments/mapanything-pip-lock.txt`](environments/mapanything-pip-lock.txt)、[`environments/mvsanywhere-pip-lock.txt`](environments/mvsanywhere-pip-lock.txt) 和两个 Conda lock 文件为准。

| 依赖 | 主要用途 | 许可证族 | 项目来源 |
|---|---|---|---|
| PyTorch | GPU 推理与张量运算 | BSD 3-Clause | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
| NumPy | 数值与数组处理 | BSD 3-Clause | [numpy/numpy](https://github.com/numpy/numpy) |
| SciPy | 稀疏图、连通性和图像处理 | BSD 3-Clause；二进制包可能包含另行许可的运行库 | [scipy/scipy](https://github.com/scipy/scipy) |
| OpenCV / opencv-python | 图像、几何和重投影处理 | Apache License 2.0 | [opencv/opencv](https://github.com/opencv/opencv) |
| Pillow | 图像读取、转换和掩码生成 | HPND | [python-pillow/Pillow](https://github.com/python-pillow/Pillow) |
| trimesh | OBJ/GLB 加载、导出和验证 | MIT | [mikedh/trimesh](https://github.com/mikedh/trimesh) |
| Open3D | 点云、TSDF 和网格处理 | MIT | [isl-org/Open3D](https://github.com/isl-org/Open3D) |
| pycolmap | COLMAP Python 接口 | BSD 3-Clause | [colmap/pycolmap](https://github.com/colmap/pycolmap) |

以上是直接依赖摘要，不是完整的传递依赖清单。部署镜像或安装包发布前，应从最终构建环境生成 Software Bill of Materials（SBOM）和对应许可证清单，而不能只依赖本文件。

## 必须保留的 MVSAnywhere 署名

使用 MVSAnywhere 的研究结果、演示或说明页面应按照其许可证要求确认作者和项目，并引用：

> MVSAnywhere: Zero Shot Multi-View Stereo. S. Izquierdo, M. Sayed, M. Firman, G. Garcia-Hernando, D. Turmukhambetov, J. Civera, O. Mac Aodha, G. Brostow, and J. Watson. CVPR 2025.

完整 BibTeX 以 [MVSAnywhere 上游 README](https://github.com/nianticlabs/mvsanywhere#bibtex) 为准。

## Web 部署时的展示要求

后续网页至少应提供一个可从页脚访问的“第三方与许可证”页面，并展示：

1. “仅用于学习、研究和非商业实践，不收费、不盈利”的用途声明；
2. MapAnything、MVSAnywhere、DINOv2、COLMAP 和 OpenMVS 的名称、来源链接与许可证；
3. MVSAnywhere 的作者确认和论文引用；
4. 当前实际使用的模型变体、模型来源 URL、下载日期、版本或 revision、SHA-256；
5. 第三方软件和模型按原作者条款提供、无额外担保的说明；
6. 本文件的链接，以及公开部署时适用的源代码或对应源码获取入口。

## 发布或部署前检查表

- [ ] 确认部署仍为非商业、无收费、无广告收入、无商业实体代用。
- [ ] 补录 MapAnything 当前权重的准确 Hugging Face 仓库、revision 和下载记录；无法证明时重新从官方来源获取。
- [ ] 不把 MVSAnywhere 源码和 hero 权重打入公开容器、安装包或下载资源；如确需分发，先逐条核对其许可证或取得许可。
- [ ] 为两个 OpenMVS vendor 目录补入与固定版本匹配的完整 AGPL v3 文本、版权声明和对应源码获取说明。
- [ ] 生成最终部署环境的 SBOM 与完整许可证清单。
- [ ] 在网页页脚和“关于”页面提供本文件及第三方来源入口。
- [ ] 用自有或已获授权的图片作为输入，并制定上传图片的保留和删除策略。

迁移和 vendor 固定仅用于同一学习工作区内复现实验，不改变任何第三方代码、模型、权重或二进制文件的权利归属。
