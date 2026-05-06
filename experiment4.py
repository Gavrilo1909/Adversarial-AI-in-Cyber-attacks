
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load and preprocess dataset

path = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(path)

print("Original shape:", df.shape)

df[" Label"] = df[" Label"].str.strip()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print("Shape after cleaning:", df.shape)

df["target"] = (df[" Label"] != "BENIGN").astype(int)

X = df.drop(columns=[" Label", "target"])
X = X.select_dtypes(include=["number"]).astype(float)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train distribution:\n", y_train.value_counts())
print("y_test distribution:\n", y_test.value_counts())


# 2. Train target models

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

svm = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", random_state=42)
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

targets = {
    "LR": lr,
    "RF": rf,
    "SVM": svm,
    "KNN": knn
}

for name, model in targets.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)

print("\nAll target models trained successfully.")


# 3. Baseline evaluation (compute once)

baseline_results = []
clean_predictions = {}
clean_accuracies = {}

for name, model in targets.items():
    print(f"Baseline prediction for {name}...")
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    clean_predictions[name] = pred
    clean_accuracies[name] = acc

    baseline_results.append({
        "target": name,
        "baseline_accuracy": acc
    })

baseline_df = pd.DataFrame(baseline_results)
print("\nBaseline Results:")
print(baseline_df.round(4))


# 4. Train Decision Tree surrogate

dt_surrogate = DecisionTreeClassifier(
    max_depth=10,
    random_state=42
)

print("\nTraining DT surrogate...")
dt_surrogate.fit(X_train, y_train)

dt_clean_pred = dt_surrogate.predict(X_test)
dt_clean_acc = accuracy_score(y_test, dt_clean_pred)

print(f"Decision Tree surrogate clean accuracy: {dt_clean_acc:.4f}")

# 5. Build DT-based adversarial direction

feature_importances = pd.Series(
    dt_surrogate.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nTop 15 important features from DT surrogate:")
print(feature_importances.head(15))

top_k = 10
top_features = feature_importances.head(top_k).index.tolist()

benign_mean = X_train[y_train == 0].mean()
attack_mean = X_train[y_train == 1].mean()

train_std = X_train.std().replace(0, 1e-6)

direction = (benign_mean - attack_mean) / train_std

direction_masked = pd.Series(0.0, index=X_train.columns)
direction_masked[top_features] = direction[top_features]

x_min = X_train.min()
x_max = X_train.max()


# 6. Generate adversarial samples

epsilons = [0.01, 0.05, 0.10, 0.20]
results = []

malicious_idx = (y_test == 1)

for epsilon in epsilons:
    print(f"\nRunning epsilon = {epsilon}")

    X_adv = X_test.copy().astype(float)

    X_adv.loc[malicious_idx, :] = (
        X_adv.loc[malicious_idx, :] +
        epsilon * direction_masked.values
    )

    X_adv = X_adv.clip(lower=x_min, upper=x_max, axis=1)

    # DT surrogate evaluation
    print("  Predicting DT...")
    dt_adv_pred = dt_surrogate.predict(X_adv)
    dt_adv_acc = accuracy_score(y_test, dt_adv_pred)
    dt_evasion = np.mean(dt_adv_pred[malicious_idx] == 0)

    results.append({
        "surrogate": "DT",
        "target": "DT",
        "epsilon": epsilon,
        "clean_accuracy": dt_clean_acc,
        "adversarial_accuracy": dt_adv_acc,
        "accuracy_drop": dt_clean_acc - dt_adv_acc,
        "evasion_rate": dt_evasion
    })

    # target evaluations
    for name, model in targets.items():
        print(f"  Predicting {name}...")
        adv_pred = model.predict(X_adv)

        clean_acc = clean_accuracies[name]
        adv_acc = accuracy_score(y_test, adv_pred)
        evasion_rate = np.mean(adv_pred[malicious_idx] == 0)

        results.append({
            "surrogate": "DT",
            "target": name,
            "epsilon": epsilon,
            "clean_accuracy": clean_acc,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": clean_acc - adv_acc,
            "evasion_rate": evasion_rate
        })

results_df = pd.DataFrame(results)

print("\nDT Surrogate Transfer Results:")
print(results_df.round(4))


# 7. Plot adversarial accuracy

plt.figure(figsize=(8, 5))

for target in results_df["target"].unique():
    subset = results_df[results_df["target"] == target]
    plt.plot(
        subset["epsilon"],
        subset["adversarial_accuracy"],
        marker="o",
        label=target
    )

plt.xlabel("Epsilon")
plt.ylabel("Adversarial Accuracy")
plt.ylim(0, 1.05)
plt.title("Decision Tree Surrogate: Transferability to Target Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 8. Plot evasion rate

plt.figure(figsize=(8, 5))

for target in results_df["target"].unique():
    subset = results_df[results_df["target"] == target]
    plt.plot(
        subset["epsilon"],
        subset["evasion_rate"],
        marker="o",
        label=target
    )

plt.xlabel("Epsilon")
plt.ylabel("Evasion Rate")
plt.ylim(0, 1.05)
plt.title("Decision Tree Surrogate: Evasion Rate on Target Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
