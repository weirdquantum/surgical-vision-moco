"""Matplotlib figures written to disk (no display needed)."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_history(history, path, best_epoch=None, title=None):
    epochs = history['epoch']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history['train_loss'], label='train')
    axes[0].plot(epochs, history['val_loss'], label='val')
    axes[0].set_ylabel('cross-entropy')
    axes[1].plot(epochs, history['train_accuracy'], label='train acc')
    axes[1].plot(epochs, history['val_accuracy'], label='val acc')
    if 'val_macro_f1' in history:
        axes[1].plot(epochs, history['val_macro_f1'], label='val macro-F1')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        if best_epoch:
            ax.axvline(best_epoch, color='gray', linestyle='--', label='selected')
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(matrix, class_names, path, title=None):
    matrix = np.asarray(matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
                           where=row_sums > 0)
    size = max(5, 0.8 * len(class_names))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    image = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if normalized[i, j] > 0.5 else 'black', fontsize=8)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    if title:
        ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curve(values, path, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(values) + 1), values)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title or '')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
