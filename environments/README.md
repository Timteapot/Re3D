# 环境说明

本目录保存成功实验环境的依赖快照：

- `mapanything-conda-lock.yml`、`mapanything-pip-lock.txt`；
- `mvsanywhere-conda-lock.yml`、`mvsanywhere-pip-lock.txt`；
- `mvsanywhere-upstream.yml`。

Conda 导出文件用于记录实际成功环境，不保证跨 CUDA、驱动和操作系统直接重建。PyTorch 的 `+cu128` 和 `+cu118` wheel 需要对应的 PyTorch wheel 索引。

推荐将可重建环境放在本项目的 `environments/runtime/` 下。该目录已被忽略，不属于需要归档或迁移的固定资源：

```powershell
cd D:\3Dreconstruction\Re3D
conda env create -p .\environments\runtime\mapanything -f .\environments\mapanything-conda-lock.yml
conda env create -p .\environments\runtime\mvsanywhere -f .\environments\mvsanywhere-conda-lock.yml
```

重建环境后应安装项目内固定源码：

```powershell
.\environments\runtime\mapanything\python.exe -m pip install -e .\vendor\map-anything
.\environments\runtime\mvsanywhere\python.exe -m pip install -e .\vendor\mvsanywhere
```

然后参考 `configs/paths.example.json` 更新 `configs/paths.local.json`，并运行：

```powershell
.\doctor.ps1
```

当前成功组合：

- MapAnything：Python 3.12、PyTorch 2.7.1+cu128；
- MVSAnywhere：Python 3.10、PyTorch 2.1.2+cu118；
- OpenMVS：2.4.0 Windows VC17 x64；
- MVSAnywhere 额外依赖本项目 `models/torch` 中的 DINOv2 源码缓存和 ViT-B/14 权重。
