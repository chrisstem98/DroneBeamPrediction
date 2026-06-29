# ============================================================
# STEP 4 - LLM Beam Prediction (Gemini API)
# ============================================================
# Install: pip install google-genai

from google import genai
from google.genai import types
import numpy as np
import pickle
import time
import re
import json

# ============================================================
# 4.1 Configure Gemini API
# ============================================================
GEMINI_API_KEY = "AIzaSyAAGmL9sKf6yJrMnH9oxayy0_gMoSs06Co"
MODEL_ID       = "gemini-2.5-flash"

client = genai.Client(api_key=GEMINI_API_KEY)

# ============================================================
# 4.2 Load Preprocessed Data
# ============================================================
with open('preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test        = data['X_test']
y_test        = data['y_test']
scaler        = data['scaler']
feature_cols  = data['feature_cols']

X_test_original = scaler.inverse_transform(X_test)

N_SAMPLES    = 100
np.random.seed(42)
indices      = np.random.choice(len(X_test), N_SAMPLES, replace=False)
X_sample     = X_test_original[indices]
y_sample     = y_test[indices]

# Build few-shot examples from training set
X_train          = data['X_train']
y_train          = data['y_train']
y2_train         = data.get('y2_train', y_train)
y3_train         = data.get('y3_train', y_train)
X_train_original = scaler.inverse_transform(X_train)
N_FEW_SHOT       = 5
np.random.seed(0)
few_shot_idx     = np.random.choice(len(X_train), N_FEW_SHOT, replace=False)
few_shot_examples = [
    {'X': X_train_original[i], 'y': int(y_train[i]), 'y2': int(y2_train[i]), 'y3': int(y3_train[i])}
    for i in few_shot_idx
]

print(f"Model          : {MODEL_ID}")
print(f"Samples        : {N_SAMPLES}")
print(f"Few-shot       : {N_FEW_SHOT} examples")
print(f"{'='*55}")

# ============================================================
# 4.3 Prompt Construction
# ============================================================
def create_prompt(sample, feature_cols, few_shot_examples=None):
    values = {col: sample[i] for i, col in enumerate(feature_cols)}

    examples_text = ""
    if few_shot_examples:
        examples_text = "Here are some examples of correct predictions:\n\n"
        for ex in few_shot_examples:
            ex_vals = {col: ex['X'][i] for i, col in enumerate(feature_cols)}
            examples_text += f"""Example:
- GPS Latitude       : {ex_vals['unit2_gps_lat']:.6f}
- GPS Longitude      : {ex_vals['unit2_gps_long']:.6f}
- Max beam power     : {ex_vals['pwr_max']:.4f}
- Mean beam power    : {ex_vals['pwr_mean']:.4f}
- Std beam power     : {ex_vals['pwr_std']:.4f}
- Median beam power  : {ex_vals['pwr_median']:.4f}
- Top-3 mean power   : {ex_vals['pwr_top3_mean']:.4f}
- Power range        : {ex_vals['pwr_range']:.4f}
- Altitude           : {ex_vals['unit2_altitude']:.2f}
- Distance to BS     : {ex_vals['unit2_distance']:.2f}
- Speed              : {ex_vals['unit2_speed']:.2f}
- Height             : {ex_vals['unit2_height']:.2f}
- Vertical speed     : {ex_vals['unit2_zspeed']:.2f}
- Pitch              : {ex_vals['unit2_pitch']:.2f}
Correct answer: {{"top1": {ex['y']}, "top2": {ex['y2']}, "top3": {ex['y3']}}}

"""

    prompt = f"""You are a wireless communication expert specializing in mmWave beam prediction for drone (UAV) communications.

A drone communicates with a base station that uses a phased array antenna with a codebook of 64 beams (indices 0 to 63).
The optimal beam index is the one that provides the highest received power for the current drone position.

Key relationships:
- Higher pwr_max indicates stronger signal in the best beam direction
- Higher pwr_range indicates a more directional channel with a clear optimal beam
- GPS position determines the geometric angle to the base station
- Altitude and distance directly affect the beam elevation angle
- Higher speed may indicate the drone is moving away from the current optimal beam

{examples_text}Now predict for this new sample:
- GPS Latitude       : {values['unit2_gps_lat']:.6f}
- GPS Longitude      : {values['unit2_gps_long']:.6f}
- Max beam power     : {values['pwr_max']:.4f}
- Mean beam power    : {values['pwr_mean']:.4f}
- Std beam power     : {values['pwr_std']:.4f}
- Median beam power  : {values['pwr_median']:.4f}
- Top-3 mean power   : {values['pwr_top3_mean']:.4f}
- Power range        : {values['pwr_range']:.4f}
- Altitude           : {values['unit2_altitude']:.2f}
- Distance to BS     : {values['unit2_distance']:.2f}
- Speed              : {values['unit2_speed']:.2f}
- Height             : {values['unit2_height']:.2f}
- Vertical speed     : {values['unit2_zspeed']:.2f}
- Pitch              : {values['unit2_pitch']:.2f}

Respond with ONLY a JSON object:
{{"top1": <best_beam_index>, "top2": <second_best>, "top3": <third_best>}}

Beam indices must be integers between 0 and 63."""
    return prompt

# ============================================================
# 4.4 Run Gemini Predictions
# ============================================================
def parse_response(response_text):
    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            top1   = max(0, min(63, int(parsed.get('top1', 0))))
            top2   = max(0, min(63, int(parsed.get('top2', 1))))
            top3   = max(0, min(63, int(parsed.get('top3', 2))))
            return [top1, top2, top3]
    except Exception:
        pass
    numbers = re.findall(r'\b([0-9]|[1-5][0-9]|6[0-3])\b', response_text)
    numbers = [int(n) for n in numbers[:3]]
    while len(numbers) < 3:
        numbers.append(0)
    return numbers

gemini_predictions = []
gemini_top3_preds  = []
correct_top1       = 0
correct_top2       = 0
correct_top3       = 0
errors             = 0

print(f"{'Sample':<8} {'True':>6} {'Pred Top-1':>10} {'Pred Top-3':>30} {'Top1 OK':>8} {'Run Acc':>8}")
print(f"{'='*75}")

for i, (sample, true_beam) in enumerate(zip(X_sample, y_sample)):
    try:
        prompt   = create_prompt(sample, feature_cols, few_shot_examples)
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        preds = parse_response(response.text)
        gemini_predictions.append(preds[0])
        gemini_top3_preds.append(preds)

        if preds[0] == true_beam:
            correct_top1 += 1
        if true_beam in preds[:2]:
            correct_top2 += 1
        if true_beam in preds[:3]:
            correct_top3 += 1

        ok      = 'YES' if preds[0] == true_beam else 'no'
        run_acc = correct_top1 / (i + 1)
        print(f"{i+1:<8} {true_beam:>6} {preds[0]:>10} {str(preds):>30} {ok:>8} {run_acc:>8.3f}")

        time.sleep(0.3)

    except Exception as e:
        print(f"  Error at sample {i}: {e}")
        gemini_predictions.append(0)
        gemini_top3_preds.append([0, 1, 2])
        errors += 1

# ============================================================
# 4.5 Compute Top-K Accuracy
# ============================================================
gemini_top1 = correct_top1 / N_SAMPLES
gemini_top2 = correct_top2 / N_SAMPLES
gemini_top3 = correct_top3 / N_SAMPLES

print(f"\n{'=' * 45}")
print("GEMINI RESULTS")
print(f"{'=' * 45}")
print(f"Top-1 Accuracy : {gemini_top1:.4f} ({gemini_top1*100:.2f}%)")
print(f"Top-2 Accuracy : {gemini_top2:.4f} ({gemini_top2*100:.2f}%)")
print(f"Top-3 Accuracy : {gemini_top3:.4f} ({gemini_top3*100:.2f}%)")
print(f"Errors         : {errors}/{N_SAMPLES}")

# ============================================================
# 4.6 Save Results
# ============================================================
gemini_results = {
    'top1'        : gemini_top1,
    'top2'        : gemini_top2,
    'top3'        : gemini_top3,
    'predictions' : gemini_predictions,
    'top3_preds'  : gemini_top3_preds,
    'y_sample'    : y_sample,
    'n_samples'   : N_SAMPLES
}
with open('gemini_results.pkl', 'wb') as f:
    pickle.dump(gemini_results, f)

print("\nStep 4 complete. Saved: gemini_results.pkl")