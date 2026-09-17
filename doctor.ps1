$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Paths = Get-Content -LiteralPath (Join-Path $Root 'configs\paths.local.json') -Raw | ConvertFrom-Json
$Python = if ($env:RE3D_MAP_PYTHON) { $env:RE3D_MAP_PYTHON } else { $Paths.mapanything_python }
& $Python (Join-Path $Root 'scripts\run_pipeline.py') --doctor
exit $LASTEXITCODE
