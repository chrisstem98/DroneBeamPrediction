# ============================================================
# STEP 3 - ML Model Training with Grid Search (RF, KNN, MLP)
# ============================================================

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# ============================================================
# 3.1 Load Preprocessed Data
# ============================================================
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train    = data['X_train']
X_val      = data['X_val']
X_test     = data['X_test']
y_train    = data['y_train']
y_val      = data['y_val']
y_test     = data['y_test']

X_trainval = np.vstack([X_train, X_val])
y_trainval = np.concatenate([y_train, y_val])

print(f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")

# ============================================================
# 3.2 Helper Functions
# ============================================================
def topk_accuracy(y_true, y_prob, k):
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    correct     = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)

def get_full_prob_matrix(model, X, n_beams=64):
    y_prob       = model.predict_proba(X)
    classes      = model.classes_
    y_prob_full  = np.zeros((len(X), n_beams))
    for i, cls in enumerate(classes):
        if cls < n_beams:
            y_prob_full[:, cls] = y_prob[:, i]
    return y_prob_full

# ============================================================
# 3.3 Grid Search
# ============================================================
print("\n" + "=" * 55)
print("GRID SEARCH HYPERPARAMETER TUNING")
print("=" * 55)

# Random Forest
print("\n[1/3] Grid Search: Random Forest...")
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
    cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
start = time.time()
rf_grid.fit(X_trainval, y_trainval)
print(f"  Best params : {rf_grid.best_params_}")
print(f"  CV accuracy : {rf_grid.best_score_:.4f}")
print(f"  Time        : {time.time()-start:.1f}s")

# KNN
print("\n[2/3] Grid Search: KNN...")
knn_grid = GridSearchCV(
    KNeighborsClassifier(n_jobs=-1),
    {'n_neighbors': [3, 5, 7, 10], 'metric': ['euclidean', 'manhattan'], 'weights': ['uniform', 'distance']},
    cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
start = time.time()
knn_grid.fit(X_trainval, y_trainval)
print(f"  Best params : {knn_grid.best_params_}")
print(f"  CV accuracy : {knn_grid.best_score_:.4f}")
print(f"  Time        : {time.time()-start:.1f}s")

# MLP
print("\n[3/3] Grid Search: MLP...")
mlp_grid = GridSearchCV(
    MLPClassifier(random_state=42),
    {'hidden_layer_sizes': [(64,), (128, 64), (256, 128)], 'learning_rate_init': [0.001, 0.01], 'max_iter': [500,1000]},
    cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
start = time.time()
mlp_grid.fit(X_trainval, y_trainval)
print(f"  Best params : {mlp_grid.best_params_}")
print(f"  CV accuracy : {mlp_grid.best_score_:.4f}")
print(f"  Time        : {time.time()-start:.1f}s")

# ============================================================
# 3.4 Evaluate on Test Set
# ============================================================
print("\n" + "=" * 55)
print("TEST SET EVALUATION")
print("=" * 55)

best_models = {
    'Random Forest': rf_grid.best_estimator_,
    'KNN'          : knn_grid.best_estimator_,
    'MLP'          : mlp_grid.best_estimator_
}

results = {}

for name, model in best_models.items():
    print(f"\nModel: {name}")

    y_prob_val  = get_full_prob_matrix(model, X_val)
    val_top1    = topk_accuracy(y_val, y_prob_val, k=1)
    val_top2    = topk_accuracy(y_val, y_prob_val, k=2)
    val_top3    = topk_accuracy(y_val, y_prob_val, k=3)
    print(f"  Validation -> Top-1: {val_top1:.4f} | Top-2: {val_top2:.4f} | Top-3: {val_top3:.4f}")

    start       = time.time()
    y_prob_test = get_full_prob_matrix(model, X_test)
    inf_time    = time.time() - start

    top1 = topk_accuracy(y_test, y_prob_test, k=1)
    top2 = topk_accuracy(y_test, y_prob_test, k=2)
    top3 = topk_accuracy(y_test, y_prob_test, k=3)
    print(f"  Test       -> Top-1: {top1:.4f} | Top-2: {top2:.4f} | Top-3: {top3:.4f}")
    print(f"  Inference time: {inf_time:.4f}s")

    results[name] = {
        'top1': top1, 'top2': top2, 'top3': top3,
        'val_top1': val_top1, 'val_top2': val_top2, 'val_top3': val_top3,
        'inference_time': inf_time,
        'y_prob_full': y_prob_test,
        'model': model
    }

# ============================================================
# 3.5 Save Results
# ============================================================
with open('ml_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# ============================================================
# 3.6 Visualization
# ============================================================
model_names = list(results.keys())
top1_vals   = [results[m]['top1'] for m in model_names]
top2_vals   = [results[m]['top2'] for m in model_names]
top3_vals   = [results[m]['top3'] for m in model_names]
val_top1    = [results[m]['val_top1'] for m in model_names]

x     = np.arange(len(model_names))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bars1 = axes[0].bar(x - width, top1_vals, width, label='Top-1', color='steelblue')
bars2 = axes[0].bar(x,         top2_vals, width, label='Top-2', color='orange')
bars3 = axes[0].bar(x + width, top3_vals, width, label='Top-3', color='green')
axes[0].set_title('Top-K Beam Prediction Accuracy (Test Set)\nwith Grid Search Hyperparameters')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names)
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim(0, 1.15)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
for bar in [*bars1, *bars2, *bars3]:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{bar.get_height():.2f}',
                 ha='center', va='bottom', fontsize=8)

axes[1].bar(x - 0.2, val_top1,  0.4, label='Validation Top-1', color='steelblue', alpha=0.7)
axes[1].bar(x + 0.2, top1_vals, 0.4, label='Test Top-1',       color='orange',    alpha=0.7)
axes[1].set_title('Validation vs Test Top-1 Accuracy')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names)
axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1.15)
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('step3_ml_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 55)
print("BEST HYPERPARAMETERS")
print("=" * 55)
print(f"Random Forest : {rf_grid.best_params_}")
print(f"KNN           : {knn_grid.best_params_}")
print(f"MLP           : {mlp_grid.best_params_}")
print("\nStep 3 complete. Saved: ml_results.pkl, step3_ml_results.png")