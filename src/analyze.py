"""
Analyze and compare all augmentation experiments.
Run: python src/analyze.py
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RESULTS_DIR = "results"
FIG_DIR = "figures"
CLASS_FILE = "data/class_list_food.txt"
os.makedirs(FIG_DIR, exist_ok=True)

EXPERIMENTS = ["baseline", "geometric", "color_jitter", "cutout", "mixup", "cutmix", "combined"]

FOOD_FAMILIES = {
    "Egg Dishes":       [31, 39, 53, 60],
    "Cakes & Desserts": [7, 20, 30, 41, 43, 54, 55, 57, 61, 63, 72, 74, 80],
    "Pasta & Rice":     [1, 27, 40, 44, 50, 51, 68, 79],
    "Meat":             [8, 13, 16, 21, 35, 62, 64, 73, 75],
    "Seafood":          [18, 26, 29, 33, 36, 37, 71],
    "Fried & Battered": [10, 15, 38, 58],
    "Sandwiches":       [2, 42, 45, 49],
    "Soups & Stews":    [34, 59, 69, 78],
    "Salads & Dips":    [4, 17, 46, 70],
    "Dumplings & Wraps":[23, 25, 47, 52, 76],
}

# Load class names
class_names = {}
with open(CLASS_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            class_names[int(parts[0])] = parts[1]

# Load results
results = {}
for exp in EXPERIMENTS:
    exp_dir = os.path.join(RESULTS_DIR, exp)
    if not os.path.exists(exp_dir):
        print(f"  Skipping {exp} (not found)")
        continue
    with open(os.path.join(exp_dir, "history.json"), "r") as f:
        history = json.load(f)
    per_class_acc = np.load(os.path.join(exp_dir, "best_per_class_acc.npy"))
    results[exp] = {
        "history": history,
        "per_class_acc": per_class_acc,
        "best_val_acc": max(history["val_acc"]),
    }
    print(f"  {exp}: best val acc = {results[exp]['best_val_acc']:.4f}")

# --- 1. Bar chart: overall accuracy ---
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]["best_val_acc"] for n in names]
colors = ["#636363" if n == "baseline" else "#FF5722" if n == "combined" else "#2196F3" for n in names]
bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=1.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Validation Accuracy")
ax.set_title("Overall Accuracy by Augmentation Strategy")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "accuracy_comparison.png"), dpi=150)
plt.close()
print("Saved accuracy_comparison.png")

# --- 2. Training curves ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, data in results.items():
    epochs = range(1, len(data["history"]["val_acc"]) + 1)
    ax1.plot(epochs, data["history"]["val_acc"], label=name)
    ax2.plot(epochs, data["history"]["train_loss"], label=name)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val Accuracy"); ax1.set_title("Validation Accuracy"); ax1.legend(); ax1.grid(alpha=0.3)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Train Loss"); ax2.set_title("Training Loss"); ax2.legend(); ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=150)
plt.close()
print("Saved training_curves.png")

# --- 3. Per-class heatmap ---
exp_names = list(results.keys())
acc_matrix = np.array([results[n]["per_class_acc"] for n in exp_names])
fig, ax = plt.subplots(figsize=(24, 8))
sns.heatmap(acc_matrix, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
            xticklabels=[class_names.get(i+1, str(i+1)) for i in range(80)],
            yticklabels=exp_names)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
ax.set_title("Per-Class Accuracy by Augmentation Strategy")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "per_class_heatmap.png"), dpi=150)
plt.close()
print("Saved per_class_heatmap.png")

# --- 4. Food family improvement heatmap (KEY POSTER FIGURE) ---
if "baseline" in results:
    baseline_acc = results["baseline"]["per_class_acc"]
    family_names = list(FOOD_FAMILIES.keys())
    aug_names = [n for n in exp_names if n != "baseline"]

    improvement = np.zeros((len(family_names), len(aug_names)))
    for i, family in enumerate(family_names):
        indices = [c - 1 for c in FOOD_FAMILIES[family]]
        base_avg = baseline_acc[indices].mean()
        for j, aug in enumerate(aug_names):
            aug_avg = results[aug]["per_class_acc"][indices].mean()
            improvement[i, j] = aug_avg - base_avg

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(improvement, ax=ax, cmap="RdYlGn", center=0, annot=True, fmt=".3f",
                xticklabels=aug_names, yticklabels=family_names)
    ax.set_title("Accuracy Change vs Baseline by Food Family")
    ax.set_xlabel("Augmentation Strategy")
    ax.set_ylabel("Food Family")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "family_improvement_heatmap.png"), dpi=150)
    plt.close()
    print("Saved family_improvement_heatmap.png")

# --- 5. Confusion matrix for baseline ---
if "baseline" in results:
    preds = np.load(os.path.join(RESULTS_DIR, "baseline", "best_preds.npy"))
    labels = np.load(os.path.join(RESULTS_DIR, "baseline", "best_labels.npy"))
    cm = confusion_matrix(labels, preds, labels=range(80))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", vmin=0, vmax=0.5,
                xticklabels=[class_names.get(i+1, str(i+1)) for i in range(80)],
                yticklabels=[class_names.get(i+1, str(i+1)) for i in range(80)])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Baseline Confusion Matrix (Normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "confusion_matrix_baseline.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix_baseline.png")

print("\nAll analysis complete!")