
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

print("Full X_train:", X_train.shape)
print("Full X_test:", X_test.shape)


# 2. Reduce test size for faster runtime

X_test, _, y_test, _ = train_test_split(
    X_test, y_test,
    train_size=20000,
    random_state=42,
    stratify=y_test
)

print("Reduced X_test:", X_test.shape)
print("Reduced y_test distribution:\n", y_test.value_counts())


# 3. Helper functions

def majority_vote(pred_matrix):
    # pred_matrix shape = (n_models, n_samples)
    votes = np.sum(pred_matrix, axis=0)
    return (votes >= (pred_matrix.shape[0] / 2)).astype(int)

def get_lr_direction(model, feature_names):
    scaler = model.named_steps["standardscaler"]
    lr_model = model.named_steps["logisticregression"]

    weights = lr_model.coef_[0]
    scales = np.where(scaler.scale_ == 0, 1e-6, scaler.scale_)

    # approximate direction in original feature space
    direction = pd.Series(np.sign(weights) / scales, index=feature_names)
    return direction

def get_centroid_direction(X_train, y_train):
    benign_mean = X_train[y_train == 0].mean()
    attack_mean = X_train[y_train == 1].mean()
    std = X_train.std().replace(0, 1e-6)

    direction = (benign_mean - attack_mean) / std
    return direction

def normalize_direction(direction):
    norm = np.linalg.norm(direction.values)
    if norm == 0:
        return direction
    return direction / norm


# 4. # Train the target ensemble models (LR and RF only for speed)


target_models = {
    "LR": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in target_models.items():
    print(f"Training target model: {name}")
    model.fit(X_train, y_train)

# Clean ensemble prediction
target_clean_preds = []
for name, model in target_models.items():
    print(f"Clean prediction for target model: {name}")
    pred = model.predict(X_test)
    target_clean_preds.append(pred)

target_clean_preds = np.vstack(target_clean_preds)
ensemble_clean_pred = majority_vote(target_clean_preds)
ensemble_clean_acc = accuracy_score(y_test, ensemble_clean_pred)

print(f"\nTarget ensemble clean accuracy: {ensemble_clean_acc:.4f}")


# 5. Train surrogate model pool once

surrogate_pool = {
    "LR": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "RF": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

for name, model in surrogate_pool.items():
    print(f"Training surrogate model: {name}")
    model.fit(X_train, y_train)


# 6. # Define surrogate configurations: fewer models vs. same models

surrogate_configs = {
    "1_model_LR": ["LR"],
    "2_models_LR_RF": ["LR", "RF"]
}

epsilons = [0.01, 0.05, 0.10]
results = []

x_min = X_train.min()
x_max = X_train.max()

centroid_direction = normalize_direction(get_centroid_direction(X_train, y_train))
malicious_idx = (y_test == 1)


# 7. Run experiment

for config_name, surrogate_names in surrogate_configs.items():
    print(f"\nRunning surrogate config: {config_name}")

    directions = []

    for model_name in surrogate_names:
        model = surrogate_pool[model_name]

        if model_name == "LR":
            dir_vec = get_lr_direction(model, X_train.columns)
            dir_vec = normalize_direction(dir_vec)
        else:
            # heuristic direction for RF
            dir_vec = centroid_direction.copy()

        directions.append(dir_vec)

    combined_direction = pd.concat(directions, axis=1).mean(axis=1)
    combined_direction = normalize_direction(combined_direction)

    for epsilon in epsilons:
        print(f"  Epsilon = {epsilon}")

        X_adv = X_test.copy().astype(float)

        # perturb only malicious samples
        X_adv.loc[malicious_idx, :] = (
            X_adv.loc[malicious_idx, :] +
            epsilon * combined_direction.values
        )

        X_adv = X_adv.clip(lower=x_min, upper=x_max, axis=1)

        # evaluate on target ensemble
        ensemble_adv_preds = []
        for target_name, target_model in target_models.items():
            print(f"    Predicting target {target_name}")
            pred = target_model.predict(X_adv)
            ensemble_adv_preds.append(pred)

        ensemble_adv_preds = np.vstack(ensemble_adv_preds)
        ensemble_adv_pred = majority_vote(ensemble_adv_preds)

        adv_acc = accuracy_score(y_test, ensemble_adv_pred)
        evasion_rate = np.mean(ensemble_adv_pred[malicious_idx] == 0)

        results.append({
            "surrogate_config": config_name,
            "num_surrogate_models": len(surrogate_names),
            "target_config": "2_model_ensemble_LR_RF",
            "epsilon": epsilon,
            "clean_accuracy": ensemble_clean_acc,
            "adversarial_accuracy": adv_acc,
            "accuracy_drop": ensemble_clean_acc - adv_acc,
            "evasion_rate": evasion_rate
        })

results_df = pd.DataFrame(results)

print("\nFinal Experiment Results:")
print(results_df.round(4))


# 8. Plot adversarial accuracy

plt.figure(figsize=(8, 5))

for config in results_df["surrogate_config"].unique():
    subset = results_df[results_df["surrogate_config"] == config]
    plt.plot(
        subset["epsilon"],
        subset["adversarial_accuracy"],
        marker="o",
        label=config
    )

plt.xlabel("Epsilon")
plt.ylabel("Adversarial Accuracy")
plt.ylim(0, 1.05)
plt.title("Effect of Surrogate Model Count on Ensemble Target Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 9. Plot evasion rate

plt.figure(figsize=(8, 5))

for config in results_df["surrogate_config"].unique():
    subset = results_df[results_df["surrogate_config"] == config]
    plt.plot(
        subset["epsilon"],
        subset["evasion_rate"],
        marker="o",
        label=config
    )

plt.xlabel("Epsilon")
plt.ylabel("Evasion Rate")
plt.ylim(0, 1.05)
plt.title("Effect of Surrogate Model Count on Ensemble Target Evasion")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
