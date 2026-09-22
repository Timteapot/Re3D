# OpenMVS 与修复版 TextureMesh

本文记录当前 `main` 的 OpenMVS 二进制边界，以及修复版 `TextureMesh` 的版本控制和恢复方法。不要把本文理解为“整套 OpenMVS 已升级”：当前仅纹理进程使用修复版，几何阶段继续使用仓库内原版本。

## 当前固定组合

| 用途 | 程序 | 位置 | Git 状态 |
|---|---|---|---|
| COLMAP 转换 | `InterfaceCOLMAP.exe` | `vendor/openmvs-2.4.0-windows/vc17/x64/Release/` | 受 Git 管理 |
| PatchMatch/DMAP 融合 | `DensifyPointCloud.exe` | 同上 | 受 Git 管理 |
| 网格重建 | `ReconstructMesh.exe` | 同上 | 受 Git 管理 |
| 网格纹理化 | `TextureMesh.exe` | `vendor/openmvs-2.4.0-33d9484-windows/vc18/x64/Release/` | 受 Git 管理 |

`scripts/run_pipeline.py` 对前三个程序使用固定的 vendor 路径。`TextureMesh` 按以下优先级解析：

1. 环境变量 `RE3D_TEXTUREMESH_EXE`；
2. `configs/paths.local.json` 的 `texturemesh_executable`；
3. 仓库内修复版 `vendor/openmvs-2.4.0-33d9484-windows/.../TextureMesh.exe`。

当前 [`configs/paths.example.json`](../../configs/paths.example.json) 显式使用同一 vendor 相对路径。`paths.local.json` 被忽略，必须由每台机器自行创建，主要用于两个 Python 环境或临时 TextureMesh 覆盖。

## 修复来源与已验证构建

OpenMVS 2.4.0 原构建在全局接缝平衡的边界情况下可能产生大面积黑纹理。上游修复为：

- 提交：[`eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d`](https://github.com/cdcseacave/openMVS/commit/eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d)
- 标题：`texture: fix corner case in global seam leveling logic`
- 关键修改：`libs/Common/Types.inl` 与 `libs/MVS/SceneTexture.cpp`

当前仓库固定的不是只打单行补丁的 2.4.0，而是包含该修复的后续官方 CI 构建：

- OpenMVS commit：`33d9484`
- GitHub Actions run：[`34769747527`](https://github.com/cdcseacave/openMVS/actions/runs/34769747527)，项目内称为 run 633
- artifact：`OpenMVS_Windows_Release_x64`
- artifact 内目录：`vc18/x64/Release`
- `TextureMesh --help` 标识：`OpenMVS x64 v2.4.0 (33d9484)`
- 构建时间：`Sep 13 2026, 16:52:28`

上游构建说明要求 Git、CMake、支持 C++17 的编译器，并通过 vcpkg 管理依赖；Windows 官方 CI 使用 x64 Release、`x64-windows-release` triplet 和 `-A x64`。本项目把当前 artifact 的 `vc18` 目录视为编译 ABI 身份的一部分。若自行复现，应使用 Visual Studio/Build Tools 18 的“使用 C++ 的桌面开发”工作负载和 x64 工具链；改用其他 MSVC 主版本时，产物目录和校验值可能不同，必须重新做兼容性测试。

## 为什么必须保留完整 Release 目录

`TextureMesh.exe` 不是独立静态程序。当前修复构建的 Release 目录共有 86 个文件，至少直接或间接依赖：

```text
TextureMesh.exe
MVS.dll
Common.dll
IO.dll
Math.dll
boost_iostreams-vc145-mt-x64-1_91.dll
boost_program_options-vc145-mt-x64-1_91.dll
boost_serialization-vc145-mt-x64-1_91.dll
opencv_core4.dll
opencv_imgcodecs4.dll
opencv_imgproc4.dll
...其余同次 artifact 的传递依赖 DLL
```

必须保留或移动整个 `vc18/x64/Release` 目录。禁止以下做法：

- 只把 `TextureMesh.exe` 复制进仓库内 VC17 Release；
- 让 VC18 `TextureMesh.exe` 从 `PATH` 偶然加载 VC17 DLL；
- 挑选少量 DLL 后删除其余文件，除非用依赖分析和目标机测试重新验证；
- 用不同提交或不同 vcpkg 解析结果生成的 DLL 替换单个文件。

几何工具在各自 VC17 进程内加载 VC17 DLL，修复版 TextureMesh 在自己的 VC18 进程内加载 VC18 DLL；两组 DLL 不应放在同一目录。

## 推荐目录结构

复用 `vendor` 结构，并按提交和编译工具链隔离两套 OpenMVS：

```text
Re3D/
├── vendor/openmvs-2.4.0-windows/vc17/x64/Release/
│   ├── InterfaceCOLMAP.exe
│   ├── DensifyPointCloud.exe
│   ├── ReconstructMesh.exe
│   └── ...VC17 DLL                         # Git 管理
├── vendor/openmvs-2.4.0-33d9484-windows/vc18/x64/Release/
│   ├── TextureMesh.exe
│   ├── MVS.dll
│   ├── Common.dll
│   ├── IO.dll
│   ├── Math.dll
│   ├── opencv_*.dll
│   ├── boost_*.dll
│   └── ...同次构建的其余文件                 # Git 管理
├── .cache/
│   └── OpenMVS_Windows_Release_x64_run633.zip # 下载缓存，Git 忽略
└── configs/
    ├── paths.example.json                   # Git 管理
    └── paths.local.json                     # Git 忽略
```

固定 Release 共 86 个文件、约 116.6 MB；其中只有 `TextureMesh.exe` 被 Re3D 调用，保留其余程序是为了完整保存官方 artifact 和 DLL 闭包。下载 ZIP、源码和构建树仍放在被 [`.gitignore`](../../.gitignore) 忽略的 `.cache/`，不重复提交压缩包。

## 从官方 artifact 恢复

正常 Git 克隆会直接恢复 vendor 中的固定 Release，不需要访问 Actions。只有目录缺失、损坏或需要独立核验来源时才执行以下步骤。GitHub Actions artifact 可能过期，且下载通常需要登录 GitHub。

1. 打开 run `34769747527`，下载 `OpenMVS_Windows_Release_x64`。
2. 将下载文件保存为 `.cache/OpenMVS_Windows_Release_x64_run633.zip`。
3. 从仓库根目录解压完整 artifact：

```powershell
New-Item -ItemType Directory -Force .\.cache | Out-Null
Expand-Archive `
  -LiteralPath .\.cache\OpenMVS_Windows_Release_x64_run633.zip `
  -DestinationPath .\.cache\openmvs-run633 `
  -Force
```

4. 确认暂存路径是 `.cache/openmvs-run633/vc18/x64/Release/TextureMesh.exe`。若下载包额外包含同名顶层目录，应调整解压目标，而不是只复制 exe。
5. 校验暂存 ZIP 和 exe。若 vendor 目标不存在，可整目录迁移；若目标已存在，不要合并覆盖，应先保留旧目录并查明差异：

```powershell
$staging = '.\.cache\openmvs-run633'
$vendorTarget = '.\vendor\openmvs-2.4.0-33d9484-windows'
if (Test-Path -LiteralPath $vendorTarget) {
  throw 'vendor 目标已存在；停止，禁止把两个 Release 合并。'
}
Move-Item -LiteralPath $staging -Destination $vendorTarget
```

6. 按“配置”和“验证”两节完成检查；如这是仓库维护操作，应将完整 vendor 目录一并提交。

### 当前已记录的 SHA-256

| 文件 | SHA-256 |
|---|---|
| `OpenMVS_Windows_Release_x64_run633.zip` | `B4475B31B9215885444AB4A22534F04FE50C2E54F3BCB95174EB31EE83E49386` |
| `vc18/x64/Release/TextureMesh.exe` | `BB04F1D062D401DF7783E5A70BC85413812CC325627B7C8CF4E035E12F920F73` |

ZIP 校验覆盖整个下载包，优先级高于只校验 exe。自行编译或 GitHub 重新打包后的哈希通常不同，哈希不同不等于一定错误，但不能再声称与当前 artifact 字节一致。

```powershell
Get-FileHash -Algorithm SHA256 `
  .\.cache\OpenMVS_Windows_Release_x64_run633.zip, `
  .\vendor\openmvs-2.4.0-33d9484-windows\vc18\x64\Release\TextureMesh.exe
```

## artifact 不可用时从源码构建

下列命令固定到当前已验证的 OpenMVS commit，并参考官方 CI 的 vcpkg 版本和 Release 参数。命令应在 Re3D 根目录的 PowerShell 中执行；下载和编译均落入已忽略的 `.cache`。

前置条件：

- Git；
- CMake；
- Visual Studio/Build Tools 18，包含“使用 C++ 的桌面开发”和 x64 MSVC 工具；
- 足够的磁盘空间和编译时间；
- 可访问 GitHub 与 vcpkg 下载源。

```powershell
$re3dRoot = (Resolve-Path .).Path
$sourceRoot = Join-Path $re3dRoot '.cache\src'
$openmvsSource = Join-Path $sourceRoot 'openMVS-33d9484'
$openmvsBuild = Join-Path $re3dRoot '.cache\build\openMVS-33d9484'
$vcpkgRoot = Join-Path $sourceRoot 'vcpkg-37bb045f'

New-Item -ItemType Directory -Force $sourceRoot | Out-Null
git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git $openmvsSource
git -C $openmvsSource checkout 33d9484
git -C $openmvsSource submodule update --init --recursive
git -C $openmvsSource merge-base --is-ancestor eeedab7 HEAD
if ($LASTEXITCODE -ne 0) { throw '所选提交不包含 eeedab7，停止构建。' }

git clone https://github.com/microsoft/vcpkg.git $vcpkgRoot
git -C $vcpkgRoot checkout 37bb045f3c7a747d3e5d1c13b6fe6a0aec4b5d00
& (Join-Path $vcpkgRoot 'bootstrap-vcpkg.bat')

cmake -S $openmvsSource -B $openmvsBuild `
  -A x64 `
  -DCMAKE_BUILD_TYPE=Release `
  -DVCPKG_ROOT=$vcpkgRoot `
  -DVCPKG_TARGET_TRIPLET=x64-windows-release `
  -DOpenMVS_USE_CUDA=OFF `
  -DOpenMVS_HEADLESS_DEBUG=ON

cmake --build $openmvsBuild --parallel 4 --config Release
ctest --test-dir $openmvsBuild --parallel 2 --build-config Release --output-on-failure
Get-ChildItem -LiteralPath (Join-Path $openmvsBuild 'bin') `
  -Filter TextureMesh.exe -File -Recurse
```

官方 CI 会把 `make/bin/**` 整体发布，而不是只发布 `TextureMesh.exe`。本地构建完成后，先在 `.cache` 构建目录完成测试；需要替换项目固定构建时，应把 `TextureMesh.exe` 所在的完整 Release 目录作为一个整体放入新的、带提交号的 `vendor/openmvs-<version>-<commit>-windows/`，更新默认路径和文档后再提交。不要把单个 exe 或 DLL 混入现有目录。

如果 `cmake -A x64` 选中的不是 VC18，应从 VS 18 Developer PowerShell 运行，或显式选择本机 `cmake --help` 列出的 VS 18 generator。不要猜测 generator 名称；不同 CMake 版本显示名称可能不同。

## 配置

首次克隆后创建本机配置：

```powershell
Copy-Item .\configs\paths.example.json .\configs\paths.local.json
```

正常克隆无需修改 `texturemesh_executable`。若要临时测试 `.cache` 中的源码构建，只修改本机文件，例如：

```json
{
  "mapanything_python": "environments/runtime/mapanything/python.exe",
  "mvsanywhere_python": "environments/runtime/mvsanywhere/python.exe",
  "texturemesh_executable": ".cache/build/openMVS-33d9484/bin/vc18/x64/Release/TextureMesh.exe"
}
```

路径相对 Re3D 根目录解析，也可以使用本机绝对路径；绝对路径只能写入被忽略的 `paths.local.json`，不能写入 `paths.example.json`。

临时覆盖：

```powershell
$env:RE3D_TEXTUREMESH_EXE = (Resolve-Path `
  '.\vendor\openmvs-2.4.0-33d9484-windows\vc18\x64\Release\TextureMesh.exe').Path
```

环境变量的优先级高于 `paths.local.json`。排查配置时先执行：

```powershell
Get-Item Env:RE3D_TEXTUREMESH_EXE -ErrorAction SilentlyContinue
```

## 空环境恢复顺序

1. 克隆 Re3D；仓库同时带有 VC17 几何工具和固定的 VC18 TextureMesh 完整 Release。
2. 按 [`environments/README.md`](../../environments/README.md) 建立两个 Python 环境。
3. 恢复 README 所列三份模型权重。
4. 从 `paths.example.json` 创建 `paths.local.json`，填写两个 Python 路径；默认 TextureMesh vendor 路径可保持不变。
5. 运行下列验证；全部通过后再开始正式实验。

## 验证命令

### 1. 检查完整运行目录

```powershell
$textureExe = (Resolve-Path `
  '.\vendor\openmvs-2.4.0-33d9484-windows\vc18\x64\Release\TextureMesh.exe').Path
$textureRelease = Split-Path -Parent $textureExe
$required = @(
  'TextureMesh.exe', 'MVS.dll', 'Common.dll', 'IO.dll', 'Math.dll',
  'opencv_core4.dll', 'opencv_imgcodecs4.dll', 'opencv_imgproc4.dll',
  'boost_iostreams-vc145-mt-x64-1_91.dll',
  'boost_program_options-vc145-mt-x64-1_91.dll',
  'boost_serialization-vc145-mt-x64-1_91.dll'
)
$missing = $required | Where-Object {
  -not (Test-Path -LiteralPath (Join-Path $textureRelease $_) -PathType Leaf)
}
if ($missing) { throw "TextureMesh Release 缺少文件: $($missing -join ', ')" }
```

此列表是最低显式检查，不代表可以删除未列出的传递依赖。

### 2. 检查程序身份与 DLL 加载

```powershell
$helpWork = '.\.cache\texturemesh-help'
New-Item -ItemType Directory -Force $helpWork | Out-Null
Push-Location $helpWork
try {
  & $textureExe --help 2>&1 | Select-Object -First 8
} finally {
  Pop-Location
}
```

当前 artifact 应显示 `OpenMVS x64 v2.4.0 (33d9484)`。使用 `.cache` 作为工作目录可避免 OpenMVS 自身生成的帮助日志污染受 Git 管理的 vendor 目录；Windows 仍会优先从 exe 所在目录加载配套 DLL。若立即报 DLL 缺失，Release 目录不完整或 DLL 搜索路径被污染。

### 3. 检查 Re3D 解析结果

```powershell
.\doctor.ps1
.\run.ps1 -Scene docs_smoke -Images D:\photos\example -Branches c -DryRun
```

`doctor.ps1` 应把前三个 OpenMVS 程序解析到 `vendor/.../vc17/...`，把 `TextureMesh` 解析到 `vendor/.../33d9484/.../vc18/...`。`-DryRun` 只验证命令编排，不执行重建，也不证明数值结果正确。

## 恢复与回退

- `.cache` 被清空：不影响仓库固定 TextureMesh；ZIP、源码与构建缓存可按需重新获取。
- vendor 固定目录缺失或损坏：优先从干净 Git 克隆恢复；也可用上述 artifact 校验并整目录恢复，禁止逐文件混合覆盖。
- artifact 过期：从 `33d9484` 源码构建；不要随意改为最新 `develop`。
- 需要临时换构建：优先设置当前 PowerShell 会话的 `RE3D_TEXTUREMESH_EXE`，验证后再写入 `paths.local.json`。
- 需要回到保守纹理参数：保持 [`configs/pipeline.json`](../../configs/pipeline.json) 中全局/局部接缝平衡为 `0`。旧 vendor TextureMesh 仍可在两项均关闭时运行，但它不是当前推荐复现路径。

## 已知复现风险

- 固定 Release 使仓库增加约 116.6 MB；后续每次替换二进制都会增加 Git 历史体积。
- GitHub Actions artifact 有保留期限且可能要求账号权限；仓库跟踪解压文件，但不重复跟踪原始 ZIP。
- `.cache`、`paths.local.json`、Python 环境和模型权重仍不受 Git 管理；固定 TextureMesh 和同次构建文件已转为受 Git 管理。
- 自行编译会受 VC18 小版本、Windows SDK、CMake、vcpkg 下载内容和上游依赖可用性影响；即使源码提交相同，二进制哈希也可能不同。
- 目前只额外记录 artifact ZIP 与 TextureMesh exe 的 SHA-256；其余 85 个文件依赖 Git 对已提交对象的完整性校验，没有独立的上游逐文件 manifest。
- 混合调用依赖 VC17 生成的 `.mvs` 能被 `33d9484` TextureMesh 读取。当前组合已经过项目场景验证，但未来更换 OpenMVS 提交仍可能发生序列化或行为变化。
- `doctor.ps1` 检查文件存在和 Python 导入，不会验证每个传递 DLL 的来源，也不会替代真实纹理化烟雾测试。

## 上游参考

- [OpenMVS 仓库](https://github.com/cdcseacave/openMVS)
- [OpenMVS Building wiki](https://github.com/cdcseacave/openMVS/wiki/Building)
- [上游接缝修复 eeedab7](https://github.com/cdcseacave/openMVS/commit/eeedab7acceae48a92ddf44eaf4eb6ffe2fef14d)
- [当前固定 Actions run 34769747527](https://github.com/cdcseacave/openMVS/actions/runs/34769747527)
