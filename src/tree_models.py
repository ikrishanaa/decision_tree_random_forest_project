import argparse
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def load_default_dataset():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

def load_csv_dataset(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    return df

def preprocess_dataset(df, target_col):
    # Drop ID/unwanted columns
    df = df.drop(columns=[col for col in df.columns if col.lower().startswith("id")], errors="ignore")

    # Encode target if categorical
    if df[target_col].dtype == object:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    return df

def train_and_evaluate(df, target_col, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    tree_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    acc_tree = accuracy_score(y_test, y_pred_tree)

    # Visualization of Decision Tree
    plt.figure(figsize=(15, 8))
    plot_tree(tree_clf, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
    plt.savefig(os.path.join(out_dir, "decision_tree.png"))
    plt.close()

    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Feature importances
    importances = rf_clf.feature_importances_
    feature_imp_df = pd.DataFrame({"feature": X.columns, "importance": importances})
    feature_imp_df = feature_imp_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_imp_df)
    plt.title("Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importances.png"))
    plt.close()

    # Cross-validation
    cv_scores_tree = cross_val_score(tree_clf, X, y, cv=5)
    cv_scores_rf = cross_val_score(rf_clf, X, y, cv=5)

    # Metrics
    metrics = {
        "decision_tree_accuracy": acc_tree,
        "random_forest_accuracy": acc_rf,
        "decision_tree_cv_mean": np.mean(cv_scores_tree),
        "random_forest_cv_mean": np.mean(cv_scores_rf),
        "classification_report_tree": classification_report(y_test, y_pred_tree, output_dict=True),
        "classification_report_rf": classification_report(y_test, y_pred_rf, output_dict=True)
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="target", help="Target column name")
    args = parser.parse_args()

    if args.csv:
        df = load_csv_dataset(args.csv, args.target)
    else:
        df = load_default_dataset()

    df = preprocess_dataset(df, args.target)
    train_and_evaluate(df, args.target)