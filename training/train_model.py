import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("dataset/triage_dataset_2000_rows.csv")
print("Dataset shape:", df.shape)
print("Class distribution:\n", df["TriageLevel"].value_counts())

# ── 2. Encode target ──────────────────────────────────────────────────────────
label_map      = {"GREEN": 0, "YELLOW": 1, "ORANGE": 2, "RED": 3}
df["severity"] = df["TriageLevel"].map(label_map)

# ── 3. Base features ──────────────────────────────────────────────────────────
X = df[["Age", "HeartRate", "Oxygen", "Temperature", "SystolicBP",
        "RespiratoryRate", "ChestPain", "BreathProblem", "Fever",
        "Diabetes", "HeartDisease"]].copy()
y = df["severity"]

# ── 4. Feature Engineering ───────────────────────────────────────────────────
X["shock_index"]       = X["HeartRate"] / X["SystolicBP"]
X["hr_oxygen_ratio"]   = X["HeartRate"] / X["Oxygen"]
X["resp_oxygen_ratio"] = X["RespiratoryRate"] / X["Oxygen"]
X["temp_hr_ratio"]     = X["Temperature"] / X["HeartRate"]
X["age_risk"]          = (X["Age"] > 60).astype(int)
X["low_oxygen"]        = (X["Oxygen"] < 92).astype(int)
X["very_low_oxygen"]   = (X["Oxygen"] < 85).astype(int)
X["high_hr"]           = (X["HeartRate"] > 100).astype(int)
X["very_high_hr"]      = (X["HeartRate"] > 130).astype(int)
X["low_bp"]            = (X["SystolicBP"] < 90).astype(int)
X["high_temp"]         = (X["Temperature"] > 101).astype(int)
X["high_resp"]         = (X["RespiratoryRate"] > 20).astype(int)
X["very_high_resp"]    = (X["RespiratoryRate"] > 30).astype(int)
X["danger_score"]      = (
    X["age_risk"] * 1      + X["low_oxygen"] * 2      + X["very_low_oxygen"] * 3 +
    X["high_hr"] * 1       + X["very_high_hr"] * 2    + X["low_bp"] * 2 +
    X["high_temp"] * 1     + X["high_resp"] * 1        + X["very_high_resp"] * 2 +
    X["ChestPain"] * 2     + X["BreathProblem"] * 2   + X["Fever"] * 1 +
    X["Diabetes"] * 1      + X["HeartDisease"] * 2
)
X["age_x_hr"]    = X["Age"] * X["HeartRate"] / 1000
X["bp_x_oxygen"] = X["SystolicBP"] * X["Oxygen"] / 10000
X["hr_x_resp"]   = X["HeartRate"] * X["RespiratoryRate"] / 1000
X["oxygen_x_bp"] = X["Oxygen"] * X["SystolicBP"] / 10000

print(f"\nTotal features: {len(X.columns)}")

# ── 5. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 6. Sample weights ─────────────────────────────────────────────────────────
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# ── 7. Model — FASTER settings ───────────────────────────────────────────────
print("\nTraining Gradient Boosting...")

model = GradientBoostingClassifier(
    n_estimators      = 200,    # reduced from 500 → much faster
    learning_rate     = 0.1,    # slightly higher → converges faster
    max_depth         = 5,
    subsample         = 0.8,
    min_samples_split = 10,
    min_samples_leaf  = 4,
    max_features      = "sqrt",
    random_state      = 42
)

# ── 8. Cross validation — 3-fold only (fast) ─────────────────────────────────
print("Running 3-Fold CV (fast)...")
cv     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
print(f"3-Fold CV F1 : {scores.mean():.4f} ± {scores.std():.4f}")

# ── 9. Final training with sample weights ─────────────────────────────────────
print("Final training...")
model.fit(X_train, y_train, sample_weight=sample_weights)

# ── 10. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
LABELS = ["GREEN", "YELLOW", "ORANGE", "RED"]

print("\n" + "="*55)
print("  Gradient Boosting — Final Results")
print("="*55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob, multi_class='ovr'):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=LABELS))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
header = f"{'':10}" + "".join(f"{l:>10}" for l in LABELS)
print(header)
for i, row in enumerate(cm):
    print(f"{LABELS[i]:10}" + "".join(f"{v:>10}" for v in row))

print("\nTop 10 Feature Importance:")
pairs = sorted(zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
for feat, imp in pairs[:10]:
    bar = "█" * int(imp * 60)
    print(f"  {feat:<25} {bar} {imp:.4f}")

# ── 11. Save ──────────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/model.pkl",    "wb") as f: pickle.dump(model, f)
with open("model/scaler.pkl",   "wb") as f: pickle.dump(StandardScaler().fit(X_train), f)
with open("model/features.pkl", "wb") as f: pickle.dump(X.columns.tolist(), f)
with open("model/label_map.pkl","wb") as f: pickle.dump(label_map, f)

print("\n✅ Model saved to model/")
print("   Now run backend: cd backend → uvicorn main:app --reload --port 8000")