# ============================================================
# STEP 6 - Final Comparison and Visualization (all 6 models)
# ============================================================

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 6.1 Load All Results
# ============================================================
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

with open('ml_results.pkl', 'rb') as f:
    ml_results = pickle.load(f)

with open('gemini_results.pkl', 'rb') as f:
    gemini_results = pickle.load(f)

with open('lora_results.pkl', 'rb') as f:
    lora_results = pickle.load(f)

with open('beamllm_results.pkl', 'rb') as f:
    beamllm_results = pickle.load(f)

with open('handover_results.pkl', 'rb') as f:
    handover_data = pickle.load(f)

handover_results = handover_data['handover_results']
sensitivity      = handover_data['sensitivity']
thresholds       = handover_data['thresholds']
H_gt             = handover_data['H_gt']
ground_truth     = handover_data['ground_truth']
all_preds        = handover_data['ml_preds_subset']
X_test           = data['X_test']
scaler           = data['scaler']

# ============================================================
# 6.2 Summary Table
# ============================================================
all_models = [
    ('KNN',           ml_results['KNN']['top1'],           ml_results['KNN']['top2'],           ml_results['KNN']['top3']),
    ('Random Forest', ml_results['Random Forest']['top1'], ml_results['Random Forest']['top2'], ml_results['Random Forest']['top3']),
    ('MLP',           ml_results['MLP']['top1'],           ml_results['MLP']['top2'],           ml_results['MLP']['top3']),
    ('Gemini',        gemini_results['top1'],               gemini_results['top2'],               gemini_results['top3']),
    ('LoRA',          lora_results['top1'],                 lora_results['top2'],                 lora_results['top3']),
    ('BeamLLM',       beamllm_results['top1'],              beamllm_results['top2'],              beamllm_results['top3']),
]

print("=" * 70)
print(f"{'SUMMARY OF RESULTS':^70}")
print("=" * 70)
print(f"\n{'Model':<20} {'Top-1':>8} {'Top-2':>8} {'Top-3':>8} {'HDA':>8}")
print("-" * 70)

model_names_all = []
top1_all = []
top2_all = []
top3_all = []
hda_all  = []

for name, t1, t2, t3 in all_models:
    hda = handover_results[name]['hda'] if name in handover_results else 0.0
    print(f"{name:<20} {t1:>8.3f} {t2:>8.3f} {t3:>8.3f} {hda:>8.3f}")
    model_names_all.append(name)
    top1_all.append(t1)
    top2_all.append(t2)
    top3_all.append(t3)
    hda_all.append(hda)

print("=" * 70)

# ============================================================
# 6.3 Main Comparison Plot (Fig. 5 style)
# ============================================================
colors_models = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown']
x     = np.arange(len(model_names_all))
width = 0.25

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1   = fig.add_subplot(gs[0, :])
bars1 = ax1.bar(x - width, top1_all, width, label='Top-1', color='steelblue')
bars2 = ax1.bar(x,          top2_all, width, label='Top-2', color='orange')
bars3 = ax1.bar(x + width,  top3_all, width, label='Top-3', color='green')
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Beam Prediction Accuracy: ML vs LLM Approaches\n(Inspired by Fig. 5 of Cheng et al., IEEE Communications Magazine, 2025)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names_all, fontsize=10)
ax1.set_ylim(0, 1.15)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add horizontal line for MLP reference
ax1.axhline(y=ml_results['MLP']['top1'], color='gray', linestyle='--', alpha=0.5, label='MLP Top-1 reference')

for bar in [*bars1, *bars2, *bars3]:
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{bar.get_height():.2f}',
             ha='center', va='bottom', fontsize=8)

# ============================================================
# 6.4 Handover Detection Metrics
# ============================================================
ax2       = fig.add_subplot(gs[1, 0])
ho_names  = list(handover_results.keys())
hda_vals  = [handover_results[m]['hda'] for m in ho_names]
far_vals  = [handover_results[m]['far'] for m in ho_names]
mdr_vals  = [handover_results[m]['mdr'] for m in ho_names]
xh        = np.arange(len(ho_names))
w         = 0.25
ax2.bar(xh - w, hda_vals, w, label='HDA', color='steelblue')
ax2.bar(xh,     far_vals, w, label='FAR', color='orange')
ax2.bar(xh + w, mdr_vals, w, label='MDR', color='red')
ax2.set_title('Handover Detection Metrics', fontsize=12)
ax2.set_xticks(xh)
ax2.set_xticklabels(ho_names, fontsize=8, rotation=15)
ax2.set_ylabel('Rate')
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# ============================================================
# 6.5 Sensitivity Analysis
# ============================================================
ax3 = fig.add_subplot(gs[1, 1])
for (name, sens), color in zip(sensitivity.items(), colors_models):
    ax3.plot(thresholds, sens['hda'], marker='o', label=name, color=color, lw=2)
ax3.set_title('Sensitivity: HDA vs Threshold', fontsize=12)
ax3.set_xlabel('Threshold (theta)')
ax3.set_ylabel('HDA')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

plt.suptitle('UAV Beam Prediction and Handover Detection\nusing Multimodal Sensing and Large Language Models',
             fontsize=14, fontweight='bold', y=1.01)
plt.savefig('step6_final_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6.6 Drone Trajectory Plot
# ============================================================
X_original = scaler.inverse_transform(X_test)
N          = len(ground_truth)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

sc = axes2[0].scatter(
    X_original[:N, 1], X_original[:N, 0],
    c=ground_truth, cmap='viridis', s=8, alpha=0.7
)
plt.colorbar(sc, ax=axes2[0], label='Beam Index')
ho_idx = np.where(H_gt == 1)[0]
axes2[0].scatter(
    X_original[ho_idx, 1], X_original[ho_idx, 0],
    c='red', s=30, marker='x', label='Handover', zorder=5
)
axes2[0].set_title('Drone Trajectory - Ground Truth Beams and Handovers')
axes2[0].set_xlabel('Longitude')
axes2[0].set_ylabel('Latitude')
axes2[0].legend()

best_ml    = max(ml_results, key=lambda m: ml_results[m]['top1'])
best_preds = all_preds[best_ml]
sc2        = axes2[1].scatter(
    X_original[:N, 1], X_original[:N, 0],
    c=best_preds, cmap='viridis', s=8, alpha=0.7
)
plt.colorbar(sc2, ax=axes2[1], label='Predicted Beam Index')
axes2[1].set_title(f'Drone Trajectory - {best_ml} Predictions')
axes2[1].set_xlabel('Longitude')
axes2[1].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('step6_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6.7 LLM Progression Plot
# ============================================================
fig3, ax = plt.subplots(figsize=(10, 5))

llm_models  = ['Gemini\n(zero-shot)', 'LoRA\n(text gen)', 'BeamLLM\n(reprogramming)']
llm_top1    = [gemini_results['top1'], lora_results['top1'], beamllm_results['top1']]
ml_baseline = ml_results['MLP']['top1']

ax.bar(llm_models, llm_top1, color=['red', 'orange', 'steelblue'], alpha=0.8)
ax.axhline(y=ml_baseline, color='green', linestyle='--', lw=2, label=f'MLP baseline ({ml_baseline:.3f})')
ax.set_title('LLM Progression: From Zero-Shot to Reprogramming')
ax.set_ylabel('Top-1 Accuracy')
ax.set_ylim(0, 0.8)
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(llm_top1):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('step6_llm_progression.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 6.8 Final Conclusions
# ============================================================
best_model = model_names_all[np.argmax(top1_all)]
print(f"\n{'=' * 55}")
print("FINAL CONCLUSIONS")
print(f"{'=' * 55}")
print(f"Best model (Top-1)         : {best_model} ({max(top1_all):.3f})")
print(f"Best LLM (Top-1)           : BeamLLM ({beamllm_results['top1']:.3f})")
print(f"Gemini zero-shot (Top-1)   : {gemini_results['top1']:.3f}")
print(f"LoRA fine-tuned (Top-1)    : {lora_results['top1']:.3f}")
print(f"BeamLLM vs MLP gap         : {abs(beamllm_results['top1'] - ml_results['MLP']['top1']):.3f}")
print(f"Best handover detection    : {max(handover_results, key=lambda m: handover_results[m]['hda'])}")
print("\nStep 6 complete. Saved: step6_final_comparison.png, step6_trajectory.png, step6_llm_progression.png")