# MoRe-AST CNN/DailyMail launcher using the project venv.
#
# Usage (from workspace root or this directory):
#   .\more_ast\experiments\cnn\run_cnn.ps1
#   .\more_ast\experiments\cnn\run_cnn.ps1 -Mode debug
#   .\more_ast\experiments\cnn\run_cnn.ps1 -Mode quick  -Steps 5
#   .\more_ast\experiments\cnn\run_cnn.ps1 -Mode standard
#
# Set your OpenRouter key before running:
#   $env:OPENROUTER_API_KEY = "sk-or-..."

param(
    [ValidateSet("debug","quick","standard","shuffled")]
    [string]$Mode  = "shuffled",
    [int]   $Steps = 0,          # 0 = use value from config.toml
    [string]$SaveDir = ""
)

$ErrorActionPreference = "Stop"

# Resolve workspace root (two levels up from this script)
$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = (Resolve-Path "$ScriptDir\..\..\..").Path
$Venv          = "$WorkspaceRoot\more-ast-env\Scripts\python.exe"

if (-not (Test-Path $Venv)) {
    Write-Error "Virtual environment not found at: $Venv"
    exit 1
}

$Args = @(
    "-m", "more_ast.experiments.cnn.run_more_ast",
    "--mode", $Mode
)
if ($Steps -gt 0)      { $Args += @("--steps",    $Steps)   }
if ($SaveDir -ne "")   { $Args += @("--save_dir", $SaveDir) }

Write-Host "Using Python: $Venv"
Write-Host "Mode: $Mode | Steps: $(if ($Steps -gt 0) { $Steps } else { 'from config' })"
Write-Host ""

Set-Location $WorkspaceRoot
& $Venv @Args
