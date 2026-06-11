# ==============================================================================
# Windows JAX Installation Script - SIMPLIFIED (No uvloop issues)
# ==============================================================================
# Run this in PowerShell (as regular user, not admin) from the EEG directory
# ==============================================================================

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "JAX EEG QNN - Windows Installation Script" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check if venv is activated
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "ERROR: Virtual environment not activated!" -ForegroundColor Red
    Write-Host "Please activate with: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "`nVirtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green

# Step 1: Upgrade pip
Write-Host "`n[1/5] Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠ Warning: pip upgrade had issues, continuing..." -ForegroundColor Yellow
}

# Step 2: Install JAX core (no CUDA required on Windows)
Write-Host "`n[2/5] Installing JAX..." -ForegroundColor Cyan
pip install "jax==0.6.2" "jaxlib==0.6.2"
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ JAX installation failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Install Flax and Optax
Write-Host "`n[3/5] Installing Flax & Optax..." -ForegroundColor Cyan
pip install "flax==0.10.7" "optax==0.2.8"
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Flax/Optax installation failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Install PennyLane (without lightning-gpu or orbax)
Write-Host "`n[4/5] Installing PennyLane & data science libraries..." -ForegroundColor Cyan
pip install "pennylane==0.42.3" "pennylane-lightning==0.42.0" `
            "torcheeg==1.1.3" "numpy==1.26.4" "pandas==2.3.3" `
            "scikit-learn==1.7.2" "scipy==1.15.3"
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ PennyLane/data science installation failed!" -ForegroundColor Red
    exit 1
}

# Step 5: Install Jupyter & visualization
Write-Host "`n[5/5] Installing Jupyter, matplotlib, seaborn..." -ForegroundColor Cyan
pip install "matplotlib==3.10.8" "seaborn==0.13.2" `
            "notebook==7.5.5" "jupyter==1.1.1" "jupyterlab==4.5.6" `
            "ipykernel>=6.5.0" "tqdm==4.67.1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Jupyter installation failed!" -ForegroundColor Red
    exit 1
}

# Step 6: Verify installation
Write-Host "`n=============================================" -ForegroundColor Green
Write-Host "VERIFICATION" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

Write-Host "`nTesting imports..." -ForegroundColor Cyan

$test_code = @"
import sys
print(f'Python: {sys.version.split()[0]}')

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
print('✓ JAX imported')

import flax
import flax.linen as nn
print('✓ Flax imported')

import optax
print('✓ Optax imported')

import pennylane as qml
print('✓ PennyLane imported')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print('✓ Data science libraries imported')

import jupyter
import notebook
print('✓ Jupyter imported')

# Test JAX devices
devices = jax.devices()
print(f'✓ JAX devices: {len(devices)} device(s)')
print(f'✓ JAX backend: {jax.default_backend()}')

# Test quantum device
dev = qml.device('lightning.qubit', wires=2)
print('✓ PennyLane quantum device ready')

print('\n✅ Installation SUCCESSFUL!')
"@

python -c $test_code

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n" -ForegroundColor Green
    Write-Host "✅ SUCCESS! Your Windows environment is ready!" -ForegroundColor Green
    Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "1. Open notebook: jupyter lab EEG_QNN_JAX.ipynb" -ForegroundColor White
    Write-Host "2. Run all cells (start from top)" -ForegroundColor White
    Write-Host "3. Training should complete in ~25 minutes" -ForegroundColor White
    Write-Host "`nNote: Using lightning.qubit (CPU-optimized quantum backend)" -ForegroundColor Yellow
    Write-Host "      Still 10-100x faster than default quantum simulator" -ForegroundColor Yellow
} else {
    Write-Host "`n❌ Installation verification failed!" -ForegroundColor Red
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Check error messages above" -ForegroundColor White
    Write-Host "2. Try: pip list | findstr /i jax" -ForegroundColor White
    Write-Host "3. Try manual install: pip install -r requirements_jax.txt" -ForegroundColor White
    exit 1
}

