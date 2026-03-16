import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ── Load everything ───────────────────────────────────────────────────────────
model      = pickle.load(open("model/model.pkl",     "rb"))
scaler     = pickle.load(open("model/scaler.pkl",    "rb"))
features   = pickle.load(open("model/features.pkl",  "rb"))
model_name = pickle.load(open("model/model_name.pkl","rb"))
label_map  = pickle.load(open("model/label_map.pkl", "rb"))

# Reverse map: 0→GREEN, 1→YELLOW, 2→ORANGE, 3→RED
idx_to_label = {v: k for k, v in label_map.items()}

# ── Rebuild test set (same seed = same split every time) ──────────────────────
df = pd.read_csv("dataset/triage_dataset_2000_rows.csv")
df["severity"] = df["TriageLevel"].map(label_map)

X = df[["Age","HeartRate","Oxygen","Temperature","SystolicBP",
        "RespiratoryRate","ChestPain","BreathProblem","Fever",
        "Diabetes","HeartDisease"]].copy()
X["shock_index"]     = X["HeartRate"] / X["SystolicBP"]
X["hr_oxygen_ratio"] = X["HeartRate"] / X["Oxygen"]
X["age_risk"]        = (X["Age"] > 60).astype(int)
y = df["severity"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_input = scaler.transform(X_test) if model_name in ["Logistic Regression", "SVM"] else X_test

y_pred = model.predict(X_input)
y_prob = model.predict_proba(X_input)

LABELS = ["GREEN", "YELLOW", "ORANGE", "RED"]

# ── Results ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"  Model   : {model_name}")
print(f"  Samples : {len(y_test)}")
print("="*55)
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1 Score : {f1_score(y_test, y_pred, average='weighted')*100:.2f}%")
print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")
print("="*55)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
header = f"{'':10}" + "".join(f"{l:>10}" for l in LABELS)
print(header)
for i, row in enumerate(cm):
    print(f"{LABELS[i]:10}" + "".join(f"{v:>10}" for v in row))

# ── Feature importance ────────────────────────────────────────────────────────
if hasattr(model, "feature_importances_"):
    print("\nFeature Importance:")
    pairs = sorted(zip(features, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in pairs:
        bar = "█" * int(imp * 50)
        print(f"  {feat:<22} {bar} {imp:.4f}")

# ── Manual patient tests ──────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Manual Prediction — 4 Custom Patients")
print("="*55)

# Format: Age, HeartRate, Oxygen, Temperature, SystolicBP,
#         RespiratoryRate, ChestPain, BreathProblem, Fever, Diabetes, HeartDisease
test_patients = [
    {"label": "Expected GREEN",  "vals": [25, 70,  98, 98.6, 120, 16, 0, 0, 0, 0, 0]},
    {"label": "Expected YELLOW", "vals": [45, 100, 94, 100.4, 105, 22, 0, 1, 1, 0, 0]},
    {"label": "Expected ORANGE", "vals": [60, 120, 90, 101.5, 88, 28, 1, 1, 1, 1, 0]},
    {"label": "Expected RED",    "vals": [75, 155, 78, 103.0, 68, 36, 1, 1, 1, 1, 1]},
]

for p in test_patients:
    v = p["vals"]
    # Base features + engineered
    row = np.array([[
        v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
        v[1] / v[4],   # shock_index
        v[1] / v[2],   # hr_oxygen_ratio
        int(v[0] > 60) # age_risk
    ]])

    if model_name in ["Logistic Regression", "SVM"]:
        row = scaler.transform(row)

    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    conf  = proba[pred] * 100

    print(f"\n  {p['label']}")
    print(f"    Age={v[0]} HR={v[1]} O2={v[2]}% Temp={v[3]}F BP={v[4]} RR={v[5]}")
    print(f"    ChestPain={v[6]} BreathProblem={v[7]} Fever={v[8]} Diabetic={v[9]} HeartDisease={v[10]}")
    print(f"    --> Prediction : {idx_to_label[pred]}  (confidence: {conf:.1f}%)")
    print(f"    --> Probs      : " + " | ".join(f"{LABELS[i]}: {p*100:.1f}%" for i, p in enumerate(proba)))