"""
STEP 2: train_model.py
──────────────────────
Trains an LSTM-based lip-reading classifier on your recorded dataset.
Run AFTER data_collector.py.

Architecture:
  Input: sequence of 30 frames × 80 lip landmark coords
       → LSTM (128 units) → LSTM (64 units)
       → Dense (64) → Dropout → Dense (num_words, softmax)

Usage:
    python train_model.py

Output:
    model/lip_model.keras   ← trained model
    model/labels.json       ← word → index mapping
    model/training_plot.png ← accuracy/loss curves
"""

import json
import os
import numpy as np

# ── Configuration ──────────────────────────────────────────────
DATASET_FILE   = "training/dataset.json"
MODEL_DIR      = "model"
MODEL_PATH     = "model/lip_model.keras"
LABELS_PATH    = "model/labels.json"
PLOT_PATH      = "model/training_plot.png"

FRAMES_PER_SAMPLE = 30      # must match data_collector.py
FEATURE_DIM       = 80      # len(LIP_LANDMARKS) * 2 = 40 * 2

EPOCHS     = 80
BATCH_SIZE = 16
VAL_SPLIT  = 0.2
# ───────────────────────────────────────────────────────────────


def load_dataset(filepath):
    """
    Load dataset JSON, pad/trim sequences to FRAMES_PER_SAMPLE,
    and return (X, y, labels) ready for training.
    """
    with open(filepath) as f:
        raw = json.load(f)

    # Filter words with at least 3 samples
    words   = sorted([w for w, samples in raw.items() if len(samples) >= 3])
    labels  = {word: idx for idx, word in enumerate(words)}
    print(f"\n  Words in dataset: {words}")

    X_list, y_list = [], []

    for word, samples in raw.items():
        if word not in labels:
            continue
        for seq in samples:
            # Each seq is a list of frames, each frame is a list of coords
            arr = np.array(seq, dtype=np.float32)

            # Pad or trim to FRAMES_PER_SAMPLE
            if arr.shape[0] < FRAMES_PER_SAMPLE:
                pad = np.zeros((FRAMES_PER_SAMPLE - arr.shape[0], FEATURE_DIM), dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:FRAMES_PER_SAMPLE]

            # Pad or trim feature dimension
            if arr.shape[1] < FEATURE_DIM:
                pad = np.zeros((arr.shape[0], FEATURE_DIM - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])
            else:
                arr = arr[:, :FEATURE_DIM]

            X_list.append(arr)
            y_list.append(labels[word])

    X = np.array(X_list, dtype=np.float32)   # (N, 30, 80)
    y = np.array(y_list, dtype=np.int32)      # (N,)

    print(f"  Dataset shape: X={X.shape}, y={y.shape}")
    print(f"  Class distribution:")
    for word, idx in labels.items():
        count = np.sum(y == idx)
        print(f"    {word:12s} → {count} samples")

    return X, y, labels


def build_model(num_classes, input_shape):
    """
    LSTM-based sequence classifier.
    Input shape: (FRAMES_PER_SAMPLE, FEATURE_DIM)
    """
    # Import here so script can still be read without TF installed
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        # Normalize inputs
        layers.LayerNormalization(),

        # First LSTM — return sequences for stacking
        layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2)
        ),

        # Second LSTM — extract final temporal representation
        layers.Bidirectional(
            layers.LSTM(64, return_sequences=False, dropout=0.2)
        ),

        # Dense classifier head
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax"),
    ], name="LipSync_LSTM")

    return model


def augment_data(X, y, factor=3):
    """
    Simple data augmentation to prevent overfitting on small datasets.
    Adds Gaussian noise and slight time-shifts.
    """
    X_aug, y_aug = [X], [y]

    for _ in range(factor):
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.01, X.shape).astype(np.float32)
        X_aug.append(X + noise)
        y_aug.append(y)

        # Time shift: roll frames by ±2
        shift = np.random.randint(-2, 3)
        X_shifted = np.roll(X, shift, axis=1)
        X_aug.append(X_shifted)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)


def plot_history(history):
    """Save training curves to PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0a0d12")

        for ax in (ax1, ax2):
            ax.set_facecolor("#111620")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2738")

        ax1.plot(history.history["accuracy"],     color="#3b82f6", label="Train")
        ax1.plot(history.history["val_accuracy"], color="#06b6d4", label="Val", linestyle="--")
        ax1.set_title("Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.legend(facecolor="#111620", labelcolor="white")

        ax2.plot(history.history["loss"],     color="#f59e0b", label="Train")
        ax2.plot(history.history["val_loss"], color="#ef4444", label="Val", linestyle="--")
        ax2.set_title("Loss")
        ax2.set_xlabel("Epoch")
        ax2.legend(facecolor="#111620", labelcolor="white")

        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=120, facecolor="#0a0d12")
        plt.close()
        print(f"  Training plot saved → {PLOT_PATH}")
    except Exception as e:
        print(f"  (Plot skipped: {e})")


def train():
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    print("="*52)
    print("  LipSync2Voice — Model Trainer")
    print("="*52)

    # ── Check dataset ──
    if not os.path.exists(DATASET_FILE):
        print(f"\n❌ Dataset not found at '{DATASET_FILE}'")
        print("   Run training/data_collector.py first!\n")
        return

    # ── Load data ──
    X, y, labels = load_dataset(DATASET_FILE)

    if len(np.unique(y)) < 2:
        print("\n❌ Need at least 2 words with data. Record more samples!\n")
        return

    num_classes = len(labels)
    print(f"\n  Classes: {num_classes}   Total samples: {len(X)}")

    # ── Augment ──
    print("  Augmenting data...")
    X, y = augment_data(X, y, factor=2)
    print(f"  After augmentation: {len(X)} samples")

    # ── One-hot encode labels ──
    y_cat = to_categorical(y, num_classes=num_classes)

    # ── Shuffle ──
    idx = np.random.permutation(len(X))
    X, y_cat = X[idx], y_cat[idx]

    # ── Build model ──
    model = build_model(num_classes, (FRAMES_PER_SAMPLE, FEATURE_DIM))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Callbacks ──
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy", patience=15,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=8, min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
    ]

    # ── Train ──
    print(f"\n  Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X, y_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save labels ──
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"\n  Labels saved → {LABELS_PATH}")

    # ── Evaluate ──
    final_acc = max(history.history["val_accuracy"])
    print(f"\n{'='*52}")
    print(f"  ✅ Training complete!")
    print(f"  Best val accuracy: {final_acc*100:.1f}%")
    print(f"  Model saved → {MODEL_PATH}")
    print(f"{'='*52}\n")

    # ── Plot ──
    plot_history(history)

    # ── Tips based on accuracy ──
    if final_acc < 0.60:
        print("  💡 Accuracy is low. Tips:")
        print("     - Record more samples (aim for 20+ per word)")
        print("     - Ensure good lighting and face the camera directly")
        print("     - Speak each word clearly and consistently")
    elif final_acc < 0.80:
        print("  💡 Good start! Record more samples for better accuracy.")
    else:
        print("  🎉 Great accuracy! Your model is ready to use.")


if __name__ == "__main__":
    train()
