# MoRe-AST CNN/DailyMail launcher via OpenRouter.
#
# Usage:
#   .\more_ast\experiments\cnn\run_cnn_openrouter.ps1
#   .\more_ast\experiments\cnn\run_cnn_openrouter.ps1 -Mode debug
#   .\more_ast\experiments\cnn\run_cnn_openrouter.ps1 -Mode quick -Steps 4
#   .\more_ast\experiments\cnn\run_cnn_openrouter.ps1 -TaskModel anthropic/claude-3.5-haiku
#
# Required:
#   $env:OPENROUTER_API_KEY = "sk-or-..."

param(
    [ValidateSet("debug","quick","standard","shuffled")]
    [string]$Mode = "quick",

    [int]$Steps = 0,

    [switch]$Resume,

    [string]$TaskModel = "",

    [string]$MetaModel = "",

    [string]$SaveDir = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = (Resolve-Path "$ScriptDir\..\..\..").Path
$Python = "$WorkspaceRoot\more-ast-env\Scripts\python.exe"
$ConfigPath = "$WorkspaceRoot\more_ast\config.openrouter.toml"
$RunModule = "more_ast.experiments.cnn.run_more_ast"

if (-not (Test-Path $Python)) {
    Write-Error "Virtual environment not found at: $Python"
    exit 1
}

if (-not (Test-Path $ConfigPath)) {
    Write-Error "OpenRouter config not found at: $ConfigPath"
    exit 1
}

if (-not $env:OPENROUTER_API_KEY) {
    Write-Error "OPENROUTER_API_KEY is not set."
    exit 1
}

if ($TaskModel -ne "") { $env:MORE_AST_TASK_MODEL = $TaskModel }
if ($MetaModel -ne "") { $env:MORE_AST_META_MODEL = $MetaModel }

$Args = @(
    "-m", $RunModule,
    "--mode", $Mode,
    "--config", $ConfigPath
)

if ($Steps -gt 0) { $Args += @("--steps", $Steps) }
if ($SaveDir -ne "") { $Args += @("--save_dir", $SaveDir) }
if ($Resume) { $Args += "--resume" }

Write-Host "Using Python: $Python"
Write-Host "Config      : $ConfigPath"
Write-Host "Mode        : $Mode"
Write-Host "Resume      : $Resume"
Write-Host "Task model  : $(if ($TaskModel) { $TaskModel } else { 'from config.openrouter.toml' })"
Write-Host "Meta model  : $(if ($MetaModel) { $MetaModel } else { 'from config.openrouter.toml' })"
Write-Host ""

Set-Location $WorkspaceRoot
& $Python @Args
