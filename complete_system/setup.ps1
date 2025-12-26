# Complete System Setup with UV
# Run this script to set up the entire environment

Write-Host "🚀 Setting up Universal Agentic Data Science System..." -ForegroundColor Cyan
Write-Host "=" * 70

# Step 1: Install uv if not already installed
Write-Host "`n📦 Step 1: Installing uv package manager..." -ForegroundColor Yellow
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    Write-Host "Installing uv..."
    Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -OutFile install-uv.ps1
    powershell -ExecutionPolicy Bypass -File install-uv.ps1
    Remove-Item install-uv.ps1
}
else {
    Write-Host "✓ uv already installed" -ForegroundColor Green
}

# Step 2: Create virtual environment
Write-Host "`n🔧 Step 2: Creating virtual environment..." -ForegroundColor Yellow
uv venv .venv
Write-Host "✓ Virtual environment created" -ForegroundColor Green

# Step 3: Activate virtual environment
Write-Host "`n⚡ Step 3: Activating environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Step 4: Install all dependencies
Write-Host "`n📚 Step 4: Installing dependencies..." -ForegroundColor Yellow
uv pip install -r requirements.txt
uv pip install tabulate  # For pandas markdown output
Write-Host "✓ All dependencies installed" -ForegroundColor Green

# Step 5: Create reference folders
Write-Host "`n📁 Step 5: Creating reference folders..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "reference" | Out-Null
New-Item -ItemType Directory -Force -Path "reference\jupyter-mcp-server" | Out-Null
New-Item -ItemType Directory -Force -Path "reference\docker-mcp-examples" | Out-Null
Write-Host "✓ Reference folders created" -ForegroundColor Green

# Step 6: Download Jupyter MCP reference
Write-Host "`n🔗 Step 6: Setting up Jupyter MCP reference..." -ForegroundColor Yellow
$gitAvailable = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $gitAvailable) {
    git clone https://github.com/datalayer/jupyter-mcp-server.git reference/jupyter-mcp-server 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Jupyter MCP reference cloned" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Git clone skipped (may already exist)" -ForegroundColor DarkYellow
    }
}
else {
    Write-Host "⚠️  Git not available, skipping clone" -ForegroundColor DarkYellow
}

# Step 7: Create .env file if not exists
Write-Host "`n🔐 Step 7: Setting up environment variables..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "⚠️  Created .env file - please add your API keys!" -ForegroundColor Yellow
    }
    else {
        Write-Host "⚠️  .env.example not found, skipping" -ForegroundColor DarkYellow
    }
}
else {
    Write-Host "✓ .env file exists" -ForegroundColor Green
}

# Step 8: Verify installation
Write-Host "`n✅ Step 8: Verifying installation..." -ForegroundColor Yellow
python -c "import dspy; import langfuse; import smolagents; print('All core packages imported successfully!')"
Write-Host "✓ Installation verified" -ForegroundColor Green

# Summary
Write-Host "`n" + "=" * 70
Write-Host "🎉 SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" * 70
Write-Host "`nNext steps:" -ForegroundColor CyanCyan
Write-Host "1. Edit .env file with your API keys"
Write-Host "2. Run tests: python tests/test_multi_dataset.py"
Write-Host "3. View results in: results/"
Write-Host "4. Check Langfuse: https://cloud.langfuse.com"
Write-Host "`nTo activate environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
