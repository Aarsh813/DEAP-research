# ============================================================================
# JAX EEG QNN - Windows Environment Setup Script
# ============================================================================
# Usage: .\setup_jax_env.ps1
# This script creates a venv and installs all JAX dependencies for Windows.
#
# KEY: Handles uvloop (not supported on Windows) by installing packages
#      with --no-deps where needed and manually adding safe dependencies.
# ============================================================================

$ErrorActionPreference = "Continue"

Write-Host ("=" * 80)
Write-Host "JAX EEG QNN - Environment Setup for Windows"
Write-Host ("=" * 80)

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $projectDir "venv"
$pythonExe = Join-Path $venvDir "Scripts\python.exe"
$pipExe = Join-Path $venvDir "Scripts\pip.exe"

# ============================================================================
# Step 1: Create or reuse venv
# ============================================================================
if (Test-Path $pythonExe) {
    Write-Host "`n[1/6] Using existing venv at: $venvDir"
} else {
    Write-Host "`n[1/6] Creating new venv at: $venvDir"
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create venv. Make sure Python 3.10+ is installed."
        exit 1
    }
}

# ============================================================================
# Step 2: Upgrade pip and install build tools
# ============================================================================
Write-Host "`n[2/6] Upgrading pip and setuptools..."
& $pipExe install --upgrade pip setuptools wheel --quiet 2>&1 | Out-Null

# ============================================================================
# Step 3: Install JAX stack (CPU-only for Windows)
# ============================================================================
Write-Host "`n[3/6] Installing JAX stack (CPU for Windows)..."
& $pipExe install jax jaxlib --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: JAX install returned non-zero exit (may be OK if deps conflict)"
}

# ============================================================================
# Step 4: Install Flax & Optax WITH uvloop workaround
# ============================================================================
# Problem: Flax/Optax depend on grain -> grain-scheduler -> uvloop
#          uvloop does NOT support Windows (build fails with RuntimeError)
# Solution: Install Flax/Optax with --no-deps, then add their safe deps manually
# ============================================================================
Write-Host "`n[4/6] Installing Flax & Optax (with uvloop Windows workaround)..."

# Install flax and optax without auto-resolving deps (avoids uvloop)
& $pipExe install --no-deps flax optax --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: flax/optax --no-deps install had issues"
}

# Manually install the safe dependencies that flax/optax need
Write-Host "  Installing Flax/Optax dependencies (skipping uvloop)..."
$flaxDeps = @(
    "absl-py",
    "msgpack",
    "rich",
    "pyyaml",
    "typing_extensions",
    "numpy",
    "jax",
    "jaxlib",
    "orbax-checkpoint",
    "tensorstore",
    "etils",
    "chex",
    "clu"
)

foreach ($dep in $flaxDeps) {
    & $pipExe install $dep --quiet 2>&1 | Out-Null
    # Some may fail (transitive uvloop deps), that's OK
}

# Also install orbax without deps if it failed above
& $pipExe install --no-deps orbax-checkpoint 2>&1 | Out-Null

# ============================================================================
# Step 5: Install PennyLane, data science stack, and torcheeg
# ============================================================================
Write-Host "`n[5/6] Installing PennyLane, data science stack, and torcheeg..."

# PennyLane (no uvloop issues)
& $pipExe install pennylane pennylane-lightning --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PennyLane with --no-deps..."
    & $pipExe install --no-deps pennylane pennylane-lightning --quiet
}

# Data science stack (no uvloop issues)
& $pipExe install numpy pandas scikit-learn scipy matplotlib seaborn tqdm --quiet 2>&1 | Out-Null

# PyTorch (required by torcheeg) + torcheeg
& $pipExe install torch --quiet 2>&1 | Out-Null
& $pipExe install lmdb mne --quiet 2>&1 | Out-Null
& $pipExe install torcheeg --no-deps --quiet 2>&1 | Out-Null

# Jupyter kernel
& $pipExe install ipykernel --quiet 2>&1 | Out-Null

# ============================================================================
# Step 6: Verify installation
# ============================================================================
Write-Host "`n[6/6] Verifying installation..."
Write-Host ""

$verifyScript = @"
import sys
print(f"Python: {sys.version.split()[0]}")
errors = []
warnings = []

try:
    import jax
    print(f"  [OK] JAX: {jax.__version__} (backend: {jax.default_backend()})")
except Exception as e:
    errors.append(f"JAX: {e}")

try:
    import flax
    print(f"  [OK] Flax: {flax.__version__}")
except Exception as e:
    errors.append(f"Flax: {e}")

try:
    import optax
    print(f"  [OK] Optax: OK")
except Exception as e:
    errors.append(f"Optax: {e}")

try:
    import pennylane as qml
    print(f"  [OK] PennyLane: {qml.__version__}")
except Exception as e:
    errors.append(f"PennyLane: {e}")

try:
    import numpy as np
    print(f"  [OK] NumPy: {np.__version__}")
except Exception as e:
    errors.append(f"NumPy: {e}")

try:
    import sklearn
    print(f"  [OK] scikit-learn: {sklearn.__version__}")
except Exception as e:
    errors.append(f"scikit-learn: {e}")

try:
    import matplotlib
    print(f"  [OK] Matplotlib: {matplotlib.__version__}")
except Exception as e:
    errors.append(f"Matplotlib: {e}")

try:
    from torcheeg.datasets import DEAPDataset
    print(f"  [OK] TorchEEG: OK")
except Exception as e:
    errors.append(f"TorchEEG: {e}")

# Check uvloop status (expected to be missing on Windows)
try:
    import uvloop
    print(f"  [OK] uvloop: {uvloop.__version__} (unexpected on Windows!)")
except ImportError:
    warnings.append("uvloop: Not installed (expected on Windows, not needed)")

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for err in errors:
        print(f"  [FAIL] {err}")
else:
    print(f"\n  All core packages installed successfully!")

if warnings:
    print(f"\nNOTES:")
    for w in warnings:
        print(f"  [SKIP] {w}")
"@

& $pythonExe -c $verifyScript

Write-Host ""
Write-Host ("=" * 80)
Write-Host "Setup complete! To use:"
Write-Host "  1. Activate venv:  .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run notebook:   jupyter lab EEG_QNN_JAX.ipynb"
Write-Host ""
Write-Host "NOTE: uvloop is intentionally skipped (not supported on Windows)."
Write-Host "      This has NO functional impact - all JAX/Flax features work without it."
Write-Host ("=" * 80)
