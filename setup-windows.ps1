param(
    [switch]$SkipNode
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Join-Path $ScriptDir "parkinsons_Motor"
$WebDir = Join-Path $ProjectDir "static\myosuite_demo"

if (-not (Test-Path $ProjectDir)) {
    Write-Error "Project directory not found: $ProjectDir"
    exit 1
}

function Ensure-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Host "uv already installed: $(uv --version)"
        return
    }

    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"

    $LocalBin = Join-Path $HOME ".local\bin"
    if (Test-Path $LocalBin) {
        $env:Path = "$LocalBin;$env:Path"
    }

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Error "uv installed but not found in PATH. Restart your terminal and rerun setup-windows.ps1."
        exit 1
    }
}

function Install-FrontendDependencies {
    if ($SkipNode) {
        Write-Host "Skipping frontend dependency installation (SkipNode enabled)."
        return
    }

    if (-not (Test-Path $WebDir)) {
        Write-Host "Skipping frontend setup: $WebDir not found."
        return
    }

    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        Write-Host "npm not found. Skipping npm install (frontend assets may already be vendored)."
        return
    }

    Push-Location $WebDir
    try {
        if (Test-Path "package-lock.json") {
            Write-Host "Installing frontend dependencies with npm ci..."
            npm ci
        }
        elseif (Test-Path "package.json") {
            Write-Host "Installing frontend dependencies with npm install..."
            npm install
        }
    }
    finally {
        Pop-Location
    }
}

Ensure-Uv

Write-Host "Syncing Python dependencies with uv..."
uv sync --project "$ProjectDir"

Install-FrontendDependencies

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Run the app with:"
Write-Host "  uv run --project `"$ProjectDir`" server"
Write-Host ""
Write-Host "Open in browser:"
Write-Host "  http://localhost:8000/web"
Write-Host "  http://localhost:8000/viewer"
