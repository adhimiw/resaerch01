# Universal Agentic Data Science System - UV Setup Script
# This script sets up the complete environment using UV package manager

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "UNIVERSAL AGENTIC DATA SCIENCE SYSTEM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create UV virtual environment
Write-Host "Step 1: Creating UV virtual environment..." -ForegroundColor Yellow
if (Test-Path ".venv") 
{
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}
else 
{
    uv venv
    Write-Host "Virtual environment created" -ForegroundColor Green
}

# Step 2: Activate virtual environment
Write-Host ""
Write-Host "Step 2: Activating virtual environment..." -ForegroundColor Yellow
& ".venv\Scripts\Activate.ps1"
Write-Host "Virtual environment activated" -ForegroundColor Green

# Step 3: Install dependencies with UV
Write-Host ""
Write-Host "Step 3: Installing dependencies with UV..." -ForegroundColor Yellow
uv pip install -r requirements.txt
Write-Host "Dependencies installed" -ForegroundColor Green

# Step 4: Install Jupyter
Write-Host ""
Write-Host "Step 4: Installing Jupyter..." -ForegroundColor Yellow
uv pip install jupyter jupyterlab ipykernel notebook
Write-Host "Jupyter installed" -ForegroundColor Green

# Step 5: Check Docker
Write-Host ""
Write-Host "Step 5: Checking Docker installation..." -ForegroundColor Yellow
try 
{
    $dockerVersion = docker --version
    Write-Host "Docker found: $dockerVersion" -ForegroundColor Green
    
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) 
    {
        Write-Host "Docker daemon is running" -ForegroundColor Green
    }
    else 
    {
        Write-Host "Docker is installed but not running. Please start Docker Desktop." -ForegroundColor Red
    }
}
catch 
{
    Write-Host "Docker not found. Install from https://www.docker.com/products/docker-desktop" -ForegroundColor Red
}

# Step 6: Setup MCP configuration
Write-Host ""
Write-Host "Step 6: Setting up MCP configuration..." -ForegroundColor Yellow
if (Test-Path "mcp_config.json") 
{
    Write-Host "MCP config already exists" -ForegroundColor Green
}
else 
{
    Write-Host "MCP config not found. Creating default configuration..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SETUP COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Start Docker Desktop if not running" -ForegroundColor White
Write-Host "2. Run: jupyter lab" -ForegroundColor White
Write-Host "3. Open: demo_notebook.ipynb" -ForegroundColor White
Write-Host "4. Test: python tests/test_multi_dataset.py" -ForegroundColor White
Write-Host ""
