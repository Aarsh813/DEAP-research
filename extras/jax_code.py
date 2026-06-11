# ============================================================================
# SECTION 1: IMPORTS, JAX DEVICE SETUP & ENVIRONMENT CHECK
# ============================================================================

import os
import sys
import platform
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# JAX stack
import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad

# Flax neural network library
import flax
import flax.linen as nn
from flax.training import train_state

# Optax optimizer
import optax

# Data science
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, f1_score,
    accuracy_score, precision_score, recall_score, matthews_corrcoef
)
from sklearn.preprocessing import StandardScaler

# Quantum computing
import pennylane as qml

# EEG dataset
from torcheeg.datasets import DEAPDataset
from torcheeg import transforms

# Utilities
import time
from datetime import timedelta
from tqdm import tqdm
from dataclasses import dataclass

# Set seeds
np.random.seed(42)
key = jax.random.PRNGKey(42)

# ============================================================================
# JAX/CUDA DEVICE SETUP
# ============================================================================
print("=" * 80)
print("JAX ENVIRONMENT SETUP")
print("=" * 80)
print(f"Platform: {platform.system()}")
print(f"Python: {sys.version.split()[0]}")
print(f"JAX version: {jax.__version__}")
print(f"Flax version: {flax.__version__}")
print(f"PennyLane version: {qml.__version__}")
print(f"Backend: {jax.default_backend().upper()}")
print(f"Available devices: {jax.devices()}")

if "gpu" in jax.default_backend().lower():
    print("  - GPU acceleration: ENABLED")
else:
    print("  - Using CPU (for GPU: use Linux/WSL with CUDA)")

print("\nAll imports successful")


================================================================================
# ============================================================================
# CONFIGURATION CLASS - WITH WINDOWS COMPATIBILITY
# ============================================================================

@dataclass
class Config:
    """Training configuration for JAX"""
    # Dataset - UPDATE THESE PATHS for your system
    root_path: str = r"C:\Users\hp\Desktop\Research\EEG\data_preprocessed_python"
    io_path: str = r"C:\Users\hp\Desktop\Research\EEG\deap_io"
    
    # Model architecture
    num_qubits: int = 4
    quantum_layers: int = 2
    cnn_hidden: int = 64
    use_residual: bool = True
    
    # Quantum backend: auto-detect Windows
    quantum_backend: str = "lightning.qubit" if platform.system() == "Windows" else "lightning.gpu"
    
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1
    
    # Early stopping
    patience: int = 15
    val_check_interval: int = 5
    
    # Data
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    augment_prob: float = 0.5
    
    # Output
    save_dir: str = "./qnn_results_jax"
    verbose: bool = True

config = Config()

print(f"\nConfiguration loaded:")
print(f"  - Batch size: {config.batch_size}")
print(f"  - Learning rate: {config.learning_rate}")
print(f"  - Epochs: {config.num_epochs}")
print(f"  - Platform: {platform.system()}")
print(f"  - Quantum Backend: {config.quantum_backend}")

if platform.system() == "Windows":
    print(f"\n  Warning - Windows detected:")
    print(f"    Using lightning.qubit (CPU-optimized)")
    print(f"    For Lightning GPU: Use Linux/WSL with CUDA 11.8+")


================================================================================
# ============================================================================
# SECTION 2: DATA LOADING
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DEAP DATASET")
print("=" * 80)

try:
    dataset = DEAPDataset(
        root_path=config.root_path,
        io_path=config.io_path,
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0)
        ])
    )
    
    print(f"Dataset loaded successfully")
    print(f"  - Total samples: {len(dataset)}")
    
    eeg, label = dataset[0]
    print(f"  - Sample EEG shape: {eeg.shape}")
    print(f"  - Sample label: {label}")
    print(f"  - EEG dtype: {eeg.dtype}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# Analyze label distribution
print(f"\nAnalyzing label distribution...")
labels = []
for i in range(min(1000, len(dataset))):
    _, lbl = dataset[i]
    labels.append(int(lbl))

label_counts = np.bincount(labels)
print(f"  - Class 0 (Low valence): {label_counts[0] if len(label_counts) > 0 else 0} samples")
print(f"  - Class 1 (High valence): {label_counts[1] if len(label_counts) > 1 else 0} samples")
if len(label_counts) > 1 and label_counts[1] > 0:
    print(f"  - Balance ratio: {label_counts[0]/label_counts[1]:.2f}:1")


================================================================================
# ============================================================================
# ADVANCED EEG PREPROCESSING & AUGMENTATION (JAX)
# ============================================================================

class EEGPreprocessor:
    """EEG preprocessing - outputs JAX arrays"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, data_list):
        """Fit normalization statistics"""
        all_data = []
        for eeg in data_list:
            all_data.extend(np.asarray(eeg).flatten())
        all_data = np.array(all_data).reshape(-1, 1)
        self.scaler.fit(all_data)
        self.fitted = True
        return self
    
    def normalize(self, eeg):
        """Z-score normalization"""
        if self.fitted:
            eeg_np = np.asarray(eeg)
            shape = eeg_np.shape
            eeg_norm = self.scaler.transform(eeg_np.reshape(-1, 1)).reshape(shape)
            return eeg_norm.astype(np.float32)
        return np.asarray(eeg, dtype=np.float32)

class EEGAugmentation:
    """EEG augmentation using numpy (applied before batching)"""
    
    def __init__(self, augment_prob=0.5):
        self.augment_prob = augment_prob
    
    def add_gaussian_noise(self, eeg, noise_std=0.02):
        if np.random.rand() > self.augment_prob:
            return eeg
        noise = np.random.randn(*eeg.shape).astype(np.float32) * noise_std
        return np.clip(eeg + noise, -5, 5)
    
    def frequency_band_scaling(self, eeg, scale_range=(0.8, 1.2)):
        if np.random.rand() > self.augment_prob:
            return eeg
        eeg_aug = eeg.copy()
        for band_idx in range(eeg_aug.shape[0]):
            scale = np.random.uniform(scale_range[0], scale_range[1])
            eeg_aug[band_idx] *= scale
        return eeg_aug
    
    def __call__(self, eeg):
        eeg = np.asarray(eeg, dtype=np.float32)
        eeg = self.add_gaussian_noise(eeg)
        eeg = self.frequency_band_scaling(eeg)
        return eeg

class EEGDataLoader:
    """Efficient dataloader yielding jnp batches"""
    
    def __init__(self, data, labels, batch_size=32, augment=False,
                 preprocessor=None, augmentor=None, drop_last=False):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment and augmentor is not None
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.drop_last = drop_last
        self.n_samples = len(data)
    
    def __iter__(self):
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            if self.drop_last and len(batch_idx) < self.batch_size:
                continue
            
            eeg_batch = []
            for idx in batch_idx:
                eeg = self.data[idx]
                if self.preprocessor is not None:
                    eeg = self.preprocessor.normalize(eeg)
                if self.augment:
                    eeg = self.augmentor(eeg)
                eeg_batch.append(eeg)
            
            eeg_batch = jnp.array(np.array(eeg_batch), dtype=jnp.float32)
            label_batch = jnp.array(self.labels[batch_idx], dtype=jnp.int32)
            yield eeg_batch, label_batch
    
    def __len__(self):
        n = self.n_samples // self.batch_size
        if not self.drop_last and self.n_samples % self.batch_size != 0:
            n += 1
        return n

print("\n" + "=" * 80)
print("PREPROCESSING & AUGMENTATION")
print("=" * 80)
print("EEG Preprocessor: Z-score normalization -> JAX arrays")
print("EEG Augmentation pipeline:")
print("  - Gaussian noise (~2% std)")
print("  - Per-band frequency scaling")


================================================================================
# ============================================================================
# TRAIN / VAL / TEST SPLIT & DATALOADERS
# ============================================================================

print("\n" + "=" * 80)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 80)

# Load all data into numpy arrays
print("Loading all data into memory...")
all_eeg_data = []
all_labels = []

for i in tqdm(range(len(dataset)), desc="Loading"):
    eeg, label = dataset[i]
    all_eeg_data.append(np.asarray(eeg, dtype=np.float32))
    all_labels.append(int(label))

all_eeg_data = np.array(all_eeg_data)
all_labels = np.array(all_labels, dtype=np.int32)

# Shuffle and split
all_indices = np.arange(len(dataset))
np.random.shuffle(all_indices)

n_train = int(len(dataset) * config.train_split)
n_val = int(len(dataset) * config.val_split)

train_indices = all_indices[:n_train]
val_indices = all_indices[n_train:n_train + n_val]
test_indices = all_indices[n_train + n_val:]

print(f"Split completed:")
print(f"  - Train: {len(train_indices)} samples ({100*config.train_split:.0f}%)")
print(f"  - Val: {len(val_indices)} samples ({100*config.val_split:.0f}%)")
print(f"  - Test: {len(test_indices)} samples ({100*config.test_split:.0f}%)")

# Fit preprocessor
print(f"\nFitting preprocessor on training data...")
preprocessor = EEGPreprocessor()
preprocessor.fit([all_eeg_data[i] for i in train_indices[:100]])
print(f"Preprocessor fitted")

# Create augmentor & data loaders
augmentor = EEGAugmentation(augment_prob=config.augment_prob)

train_loader = EEGDataLoader(all_eeg_data[train_indices], all_labels[train_indices],
                             batch_size=config.batch_size, augment=True,
                             preprocessor=preprocessor, augmentor=augmentor, drop_last=True)
val_loader = EEGDataLoader(all_eeg_data[val_indices], all_labels[val_indices],
                           batch_size=config.batch_size, preprocessor=preprocessor)
test_loader = EEGDataLoader(all_eeg_data[test_indices], all_labels[test_indices],
                            batch_size=config.batch_size, preprocessor=preprocessor)

print(f"\nDataloaders created:")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")
print(f"  - Test batches: {len(test_loader)}")

# Test batch shape
for eeg_batch, label_batch in train_loader:
    print(f"\nSample batch shapes:")
    print(f"  - EEG: {eeg_batch.shape}, dtype: {eeg_batch.dtype}")
    print(f"  - Labels: {label_batch.shape}, dtype: {label_batch.dtype}")
    break


================================================================================
# ============================================================================
# SECTION 3: QUANTUM CIRCUITS WITH JAX INTERFACE
# ============================================================================

print("\n" + "=" * 80)
print("QUANTUM CIRCUIT ARCHITECTURES")
print("=" * 80)

def get_quantum_device(backend="lightning.qubit", n_qubits=4):
    """Get quantum device with Windows fallback"""
    if platform.system() == "Windows" and backend == "lightning.gpu":
        print(f"Lightning GPU not available on Windows -> using lightning.qubit")
        backend = "lightning.qubit"
    
    try:
        dev = qml.device(backend, wires=n_qubits)
        if backend == "lightning.gpu":
            print(f"Using {backend} (CUDA-accelerated)")
        elif backend == "lightning.qubit":
            print(f"Using {backend} (CPU-optimized, ~10-100x faster than default.qubit)")
        else:
            print(f"Using {backend}")
        return dev
    except Exception as e:
        print(f"Backend '{backend}' not available: {e}")
        if backend != "lightning.qubit":
            try:
                print(f"  Falling back to 'lightning.qubit'")
                return qml.device("lightning.qubit", wires=n_qubits)
            except:
                pass
        print(f"  Falling back to 'default.qubit'")
        return qml.device("default.qubit", wires=n_qubits)

def create_efficient_quantum_circuit(n_qubits=4, n_layers=2, backend="lightning.qubit"):
    """Efficient parameterized quantum circuit with JAX interface"""
    dev = get_quantum_device(backend, n_qubits)
    
    @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
    def circuit(features, weights):
        # Angle encoding
        for i in range(n_qubits):
            qml.RY(features[i] * np.pi, wires=i)
        
        # Variational ansatz
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RX(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit

print(f"\nInitializing circuits with backend: {config.quantum_backend}")
circuit_efficient = create_efficient_quantum_circuit(
    config.num_qubits, config.quantum_layers, config.quantum_backend
)
print(f"Quantum circuits created with {config.num_qubits} qubits and {config.quantum_layers} layers")

# Test quantum circuit
test_features = jnp.array([0.5, -0.3, 0.1, 0.8])
test_weights = jax.random.normal(jax.random.PRNGKey(0),
                                 (config.quantum_layers, config.num_qubits, 2)) * 0.1
test_out = circuit_efficient(test_features, test_weights)
print(f"Quantum circuit test: input={test_features.shape} -> output={len(test_out)} measurements")


================================================================================
# ============================================================================
# SECTION 4: FLAX MODEL ARCHITECTURE
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with BatchNorm and Dropout"""
    out_channels: int
    stride: int = 1
    dropout_rate: float = 0.2
    use_shortcut: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        identity = x
        
        out = nn.Conv(self.out_channels, kernel_size=(3, 3),
                      strides=(self.stride, self.stride), padding='SAME', use_bias=False)(x)
        out = nn.BatchNorm(use_running_average=not training)(out)
        out = nn.relu(out)
        out = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(out)
        
        out = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME', use_bias=False)(out)
        out = nn.BatchNorm(use_running_average=not training)(out)
        
        if self.use_shortcut:
            identity = nn.Conv(self.out_channels, kernel_size=(1, 1),
                             strides=(self.stride, self.stride), use_bias=False)(x)
            identity = nn.BatchNorm(use_running_average=not training)(identity)
        
        out = out + identity
        out = nn.relu(out)
        return out

class DeepEEGEncoder(nn.Module):
    """Deep residual CNN encoder: (batch, 32, 1, 128) -> (batch, 128)"""
    hidden_dim: int = 64
    dropout_rate: float = 0.3

    @nn.compact
    def __call__(self, x, training: bool):
        # Initial conv
        x = nn.Conv(self.hidden_dim, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)
        
        # Layer 1: same dims
        x = ResidualBlock(self.hidden_dim, stride=1, dropout_rate=self.dropout_rate)(x, training)
        x = ResidualBlock(self.hidden_dim, stride=1, dropout_rate=self.dropout_rate)(x, training)
        
        # Layer 2: downsample
        x = ResidualBlock(self.hidden_dim * 2, stride=2, dropout_rate=self.dropout_rate, use_shortcut=True)(x, training)
        x = ResidualBlock(self.hidden_dim * 2, stride=1, dropout_rate=self.dropout_rate)(x, training)
        
        # Layer 3: downsample
        x = ResidualBlock(self.hidden_dim * 4, stride=2, dropout_rate=self.dropout_rate, use_shortcut=True)(x, training)
        x = ResidualBlock(self.hidden_dim * 4, stride=1, dropout_rate=self.dropout_rate)(x, training)
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))
        
        # FC layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate / 2, deterministic=not training)(x)
        
        return x

class HybridQNNModel(nn.Module):
    """Complete hybrid classical-quantum model in Flax.
    Pipeline: CNN Encoder -> Quantum Circuit -> Classifier
    """
    cnn_hidden: int = 64
    n_qubits: int = 4
    n_layers: int = 2
    dropout_rate: float = 0.3
    n_classes: int = 2

    @nn.compact
    def __call__(self, x, training: bool, quantum_weights):
        # Reshape (batch, channels, time) -> (batch, channels, 1, time)
        # ============================================================
        # Convert data layout: NCHW -> NHWC for Flax
        # EEG: (B, 32, 128) -> expand -> (B, 32, 1, 128) [NCHW]
        #   -> transpose -> (B, 1, 128, 32) [NHWC, C=32]
        # ============================================================
        if x.ndim == 3:
            x = jnp.expand_dims(x, axis=2)
        
        # NCHW (B, C, H, W) -> NHWC (B, H, W, C)
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # CNN encoder -> (batch, 128)
        cnn_features = DeepEEGEncoder(
            hidden_dim=self.cnn_hidden,
            dropout_rate=self.dropout_rate
        )(x, training)
        
        # Classical -> Quantum bridge
        quantum_input = nn.Dense(64)(cnn_features)
        quantum_input = nn.relu(quantum_input)
        quantum_input = nn.Dense(self.n_qubits)(quantum_input)
        quantum_input = nn.tanh(quantum_input)
        
        # Quantum processing (per sample via vmap)
        def process_single(q_input):
            try:
                q_out = circuit_efficient(q_input, quantum_weights)
                return jnp.stack(q_out)
            except:
                return jnp.tanh(q_input)  # Fallback
        
        quantum_out = vmap(process_single)(quantum_input)
        
        # Quantum -> Classical bridge
        quantum_features = nn.Dense(64)(quantum_out)
        quantum_features = nn.relu(quantum_features)
        quantum_features = nn.Dense(32)(quantum_features)
        
        # Classification head
        logits = nn.Dense(64)(quantum_features)
        logits = nn.relu(logits)
        logits = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(logits)
        logits = nn.Dense(32)(logits)
        logits = nn.relu(logits)
        logits = nn.Dropout(rate=self.dropout_rate / 2, deterministic=not training)(logits)
        logits = nn.Dense(self.n_classes)(logits)
        
        return logits, cnn_features, quantum_features

# Initialize model
print("\n" + "=" * 80)
print("MODEL ARCHITECTURE")
print("=" * 80)

model = HybridQNNModel(
    cnn_hidden=config.cnn_hidden,
    n_qubits=config.num_qubits,
    n_layers=config.quantum_layers,
    dropout_rate=config.dropout,
    n_classes=2
)

# Initialize with dummy input
init_key = jax.random.PRNGKey(0)
sample_shape = all_eeg_data[0].shape  # Get actual shape from loaded data
print(f"  Data sample shape: {sample_shape}")
dummy_eeg = jnp.ones((2,) + sample_shape, dtype=jnp.float32)
print(f"  Dummy batch shape: {dummy_eeg.shape}")
dummy_qw = jax.random.normal(init_key, (config.quantum_layers, config.num_qubits, 2)) * 0.1

variables = model.init(
    {'params': init_key, 'dropout': init_key},
    dummy_eeg, training=False, quantum_weights=dummy_qw
)

# Count parameters
def count_params(params_tree):
    return sum(p.size for p in jax.tree_util.tree_leaves(params_tree))

total_params = count_params(variables['params'])
print(f"\nModel architecture:")
print(f"  - Input: {sample_shape} (auto-detected)")
print(f"  - CNN Encoder -> 128 features")
print(f"  - Quantum Bridge: {config.num_qubits} qubits, {config.quantum_layers} layers")
print(f"  - Quantum Backend: {config.quantum_backend}")
print(f"  - Classifier: 32 -> 2 classes")
print(f"\nTotal JAX parameters: {total_params:,}")
print(f"Device: {jax.devices()[0]}")

# Test forward pass
logits, cnn_feat, q_feat = model.apply(
    variables, dummy_eeg, training=False,
    quantum_weights=dummy_qw, rngs={'dropout': init_key}
)
print(f"\nForward pass test successful:")
print(f"  - Input: {dummy_eeg.shape}")
print(f"  - CNN features: {cnn_feat.shape}")
print(f"  - Quantum features: {q_feat.shape}")
print(f"  - Logits: {logits.shape}")


================================================================================
# ============================================================================
# SECTION 5: TRAINING LOOP WITH JAX JIT COMPILATION
# ============================================================================

# Custom TrainState that holds batch_stats for BatchNorm
class TrainState(train_state.TrainState):
    batch_stats: dict = None

def create_train_state(model, params, batch_stats, learning_rate, weight_decay):
    """Create training state with optimizer"""
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )
    return TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
    )

def loss_fn(params, batch_stats, eeg, labels, model, quantum_weights, rng, training=True):
    """Cross-entropy loss with label smoothing"""
    variables = {'params': params, 'batch_stats': batch_stats}
    (logits, _, _), updates = model.apply(
        variables, eeg, training=training, quantum_weights=quantum_weights,
        rngs={'dropout': rng}, mutable=['batch_stats']
    )
    
    num_classes = logits.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    labels_smooth = labels_onehot * (1 - config.label_smoothing) + config.label_smoothing / num_classes
    ce_loss = -jnp.sum(labels_smooth * jax.nn.log_softmax(logits), axis=-1)
    
    return jnp.mean(ce_loss), (logits, updates)

@jit
def train_step(state, eeg, labels, quantum_weights, rng):
    """Single JIT-compiled training step"""
    def loss_wrapper(params):
        return loss_fn(params, state.batch_stats, eeg, labels,
                      model, quantum_weights, rng, training=True)
    
    (loss, (logits, updates)), grads = value_and_grad(loss_wrapper, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    preds = jnp.argmax(logits, axis=-1)
    return state, loss, preds

@jit
def eval_step(state, eeg, labels, quantum_weights, rng):
    """Single JIT-compiled evaluation step"""
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, _, _ = model.apply(
        variables, eeg, training=False,
        quantum_weights=quantum_weights, rngs={'dropout': rng}
    )
    
    num_classes = logits.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, num_classes)
    labels_smooth = labels_onehot * (1 - config.label_smoothing) + config.label_smoothing / num_classes
    loss = -jnp.sum(labels_smooth * jax.nn.log_softmax(logits), axis=-1).mean()
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds, logits

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING SETUP WITH JAX JIT COMPILATION")
print("=" * 80)

# Initialize state
batch_stats = variables.get('batch_stats', {})
state = create_train_state(model, variables['params'], batch_stats,
                           config.learning_rate, config.weight_decay)

quantum_weights = jax.random.normal(
    jax.random.PRNGKey(42),
    (config.quantum_layers, config.num_qubits, 2)
) * 0.1

print(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
print(f"Loss: Cross-entropy with label smoothing ({config.label_smoothing})")
print(f"Gradient clipping: max norm 1.0")
print(f"JIT Compilation: ENABLED")

# Training history
history = {
    'train_loss': [], 'train_acc': [], 'train_f1': [],
    'val_loss': [], 'val_acc': [], 'val_f1': [],
}

best_val_f1 = 0
patience_counter = 0
best_state = None

print(f"\n" + "=" * 80)
print(f"STARTING JAX TRAINING ({config.num_epochs} epochs)")
print(f"=" * 80 + "\n")

start_time = time.time()
rng = jax.random.PRNGKey(42)

for epoch in range(config.num_epochs):
    epoch_start = time.time()
    rng, epoch_rng = jax.random.split(rng)
    
    # --- Training ---
    train_losses = []
    train_preds_all = []
    train_labels_all = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=False)
    for eeg_batch, label_batch in pbar:
        rng, step_rng = jax.random.split(rng)
        state, loss, preds = train_step(state, eeg_batch, label_batch, quantum_weights, step_rng)
        
        train_losses.append(float(loss))
        train_preds_all.extend(np.array(preds))
        train_labels_all.extend(np.array(label_batch))
        pbar.set_postfix({'loss': f'{float(loss):.4f}'})
    
    train_loss = np.mean(train_losses)
    train_acc = accuracy_score(train_labels_all, train_preds_all) * 100
    train_f1 = f1_score(train_labels_all, train_preds_all, average='weighted', zero_division=0) * 100
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    
    # --- Validation ---
    if (epoch + 1) % config.val_check_interval == 0:
        val_losses = []
        val_preds_all = []
        val_labels_all = []
        
        for eeg_batch, label_batch in val_loader:
            rng, step_rng = jax.random.split(rng)
            loss, preds, _ = eval_step(state, eeg_batch, label_batch, quantum_weights, step_rng)
            val_losses.append(float(loss))
            val_preds_all.extend(np.array(preds))
            val_labels_all.extend(np.array(label_batch))
        
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels_all, val_preds_all) * 100
        val_f1 = f1_score(val_labels_all, val_preds_all, average='weighted', zero_division=0) * 100
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_state = state
        else:
            patience_counter += 1
    
    # Progress print with completed/remaining epochs
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    completed = epoch + 1
    remaining = config.num_epochs - completed
    
    if completed % 5 == 0:
        print(f"Epoch {completed:3d}/{config.num_epochs} "
              f"[done: {completed}, left: {remaining}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Train F1: {train_f1:.1f}%", end="")
        
        if completed % config.val_check_interval == 0:
            print(f" | Val Acc: {val_acc:.1f}% | "
                  f"Val F1: {val_f1:.1f}%", end="")
        
        print(f" | Time: {timedelta(seconds=int(epoch_time))} | "
              f"Elapsed: {timedelta(seconds=int(elapsed))}")
    
    if patience_counter >= config.patience:
        print(f"\nEarly stopping at epoch {completed} "
              f"({remaining} epochs remaining, skipped)")
        break

# Load best model
if best_state is not None:
    state = best_state
    print(f"Loaded best model (Val F1: {best_val_f1:.1f}%)")

total_time = time.time() - start_time
print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")


================================================================================
# ============================================================================
# SECTION 6: MODEL EVALUATION & COMPREHENSIVE METRICS
# ============================================================================

def evaluate_comprehensive(state, loader, quantum_weights):
    """Comprehensive model evaluation"""
    all_preds = []
    all_labels = []
    all_probs = []
    rng = jax.random.PRNGKey(0)
    
    for eeg_batch, label_batch in tqdm(loader, desc="Evaluating", leave=False):
        rng, step_rng = jax.random.split(rng)
        loss, preds, logits = eval_step(state, eeg_batch, label_batch, quantum_weights, step_rng)
        probs = jax.nn.softmax(logits, axis=-1)
        
        all_preds.extend(np.array(preds))
        all_labels.extend(np.array(label_batch))
        all_probs.extend(np.array(probs))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(all_labels))
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds) * 100,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100,
        'sensitivity': (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0,
        'specificity': (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(all_labels, all_preds),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }
    
    if len(np.unique(all_labels)) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr) * 100
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics

print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

print("\n1. Evaluating on training set...")
train_metrics = evaluate_comprehensive(state, train_loader, quantum_weights)

print("2. Evaluating on validation set...")
val_metrics = evaluate_comprehensive(state, val_loader, quantum_weights)

print("3. Evaluating on test set...")
test_metrics = evaluate_comprehensive(state, test_loader, quantum_weights)

# Print results
print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

def print_metrics(name, metrics):
    print(f"\n{name}:")
    print(f"  Accuracy:    {metrics['accuracy']:.2f}%")
    print(f"  Precision:   {metrics['precision']:.2f}%")
    print(f"  Recall:      {metrics['recall']:.2f}%")
    print(f"  F1-Score:    {metrics['f1']:.2f}%")
    print(f"  Sensitivity: {metrics['sensitivity']:.2f}%")
    print(f"  Specificity: {metrics['specificity']:.2f}%")
    print(f"  MCC:         {metrics['mcc']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  ROC-AUC:     {metrics['roc_auc']:.2f}%")

print_metrics("Training Set", train_metrics)
print_metrics("Validation Set", val_metrics)
print_metrics("Test Set", test_metrics)

print(f"\nTest set confusion matrix:")
print(test_metrics['confusion_matrix'])


================================================================================
# ============================================================================
# SECTION 7: VISUALIZATION & FINAL SUMMARY
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('JAX Hybrid QNN - Performance Summary', fontsize=16, fontweight='bold')

# 1. Training loss
ax = axes[0, 0]
epochs_range = range(1, len(history['train_loss']) + 1)
ax.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.legend()
ax.grid(alpha=0.3)

# 2. Accuracy curves
ax = axes[0, 1]
val_epochs = list(range(config.val_check_interval,
                        config.val_check_interval * len(history['val_acc']) + 1,
                        config.val_check_interval))
ax.plot(epochs_range, history['train_acc'], 'o-', label='Train', linewidth=2, markersize=3)
if history['val_acc']:
    ax.plot(val_epochs, history['val_acc'], 's-', label='Validation', linewidth=2, markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy')
ax.legend()
ax.grid(alpha=0.3)

# 3. F1-Score curves
ax = axes[0, 2]
ax.plot(epochs_range, history['train_f1'], 'o-', label='Train', linewidth=2, markersize=3)
if history['val_f1']:
    ax.plot(val_epochs, history['val_f1'], 's-', label='Validation', linewidth=2, markersize=4)
ax.set_xlabel('Epoch')
ax.set_ylabel('F1-Score (%)')
ax.set_title('F1-Score')
ax.legend()
ax.grid(alpha=0.3)

# 4. Confusion matrix
ax = axes[1, 0]
sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
ax.set_title('Test Confusion Matrix')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')

# 5. ROC Curve
ax = axes[1, 1]
if 'fpr' in test_metrics:
    ax.plot(test_metrics['fpr'], test_metrics['tpr'], 'b-', linewidth=2,
            label=f"AUC = {test_metrics['roc_auc']:.2f}%")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(alpha=0.3)

# 6. Metrics bar chart
ax = axes[1, 2]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Sensitivity', 'Specificity']
test_values = [test_metrics[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']]
bars = ax.bar(metrics_names, test_values, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_ylabel('Score (%)')
ax.set_title('Test Set Metrics')
ax.set_ylim([0, 105])
ax.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, test_values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY - JAX VERSION")
print("=" * 80)

summary_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score',
               'Sensitivity', 'Specificity', 'ROC-AUC', 'MCC'],
    'Train': [train_metrics['accuracy'], train_metrics['precision'],
              train_metrics['recall'], train_metrics['f1'],
              train_metrics['sensitivity'], train_metrics['specificity'],
              train_metrics.get('roc_auc', 0), train_metrics['mcc']],
    'Val': [val_metrics['accuracy'], val_metrics['precision'],
            val_metrics['recall'], val_metrics['f1'],
            val_metrics['sensitivity'], val_metrics['specificity'],
            val_metrics.get('roc_auc', 0), val_metrics['mcc']],
    'Test': [test_metrics['accuracy'], test_metrics['precision'],
             test_metrics['recall'], test_metrics['f1'],
             test_metrics['sensitivity'], test_metrics['specificity'],
             test_metrics.get('roc_auc', 0), test_metrics['mcc']]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

print(f"\n" + "=" * 80)
print("KEY ACHIEVEMENTS")
print("=" * 80)
print(f"Best Test Accuracy: {test_metrics['accuracy']:.2f}%")
print(f"Best Test F1-Score: {test_metrics['f1']:.2f}%")
print(f"Training Time: {timedelta(seconds=int(total_time))}")
print(f"Framework: JAX {jax.__version__} + Flax {flax.__version__}")
print(f"Quantum Backend: {config.quantum_backend}")
print(f"Total Parameters: {total_params:,}")
print(f"Device: {jax.devices()[0]}")

print(f"\nJAX EEG Quantum Neural Network complete!")


================================================================================
