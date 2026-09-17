param(
    [string]$Scene,
    [string]$Images,
    [ValidateSet('all','a','b','c','a,b','a,c','b,c','a,b,c')]
    [string]$Branches = 'all',
    [switch]$DryRun,
    [switch]$Doctor
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Paths = Get-Content -LiteralPath (Join-Path $Root 'configs\paths.local.json') -Raw | ConvertFrom-Json
$Python = if ($env:RE3D_MAP_PYTHON) { $env:RE3D_MAP_PYTHON } else { $Paths.mapanything_python }
$Arguments = @((Join-Path $Root 'scripts\run_pipeline.py'))
if ($Doctor) {
    $Arguments += '--doctor'
} else {
    if (-not $Scene) {
        throw 'Scene is required unless -Doctor is specified.'
    }
    $Arguments += @('--scene', $Scene, '--branches', $Branches)
    if ($Images) { $Arguments += @('--images', $Images) }
    if ($DryRun) { $Arguments += '--dry-run' }
}
& $Python @Arguments
exit $LASTEXITCODE
