# Re-import libraries after code execution environment reset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline

# Re-create required directory
os.makedirs("results/figures", exist_ok=True)

# Reload uploaded data
df = pd.read_excel("data/extinction_data.xlsx", sheet_name="all_data")
df.dropna(inplace=True)

# Features and target
X = df[["f_crack", "u_fuel", "u_air", "width"]]
y = df["stable"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Models and hyperparameters
models = {
    "LogisticRegression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(solver='liblinear'))
    ]),
    "DecisionTree": Pipeline([
        ('clf', DecisionTreeClassifier())
    ]),
    "ExtraTrees": Pipeline([
        ('clf', ExtraTreesClassifier(n_jobs=-1))
    ]),
    "GradientBoosting": Pipeline([
        ('clf', GradientBoostingClassifier())
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True))
    ]),
    "GPR": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianProcessClassifier())
    ]),
    "NeuralNet": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MLPClassifier(max_iter=1000))
    ])
}

param_grids = {
    "LogisticRegression": {},
    "DecisionTree": {
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_split": [2, 5, 10]
    },
    "ExtraTrees": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [None, 10, 20]
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.1, 0.05],
        "clf__max_depth": [3, 5]
    },
    "SVM": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["rbf", "linear"]
    },
    "GPR": {},
    "NeuralNet": {
        "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "clf__alpha": [0.0001, 0.001],
        "clf__learning_rate_init": [0.001, 0.01]
    }
}

# Train and evaluate models
results = []
best_models = {}

for name, model in models.items():
    start_time = time.perf_counter()
    if param_grids[name]:
        search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
    duration = time.perf_counter() - start_time

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report["accuracy"]
    f1 = report["1"]["f1-score"]

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-score": f1,
        "Train Time (s)": duration
    })

    best_models[name] = best_model

    disp = ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    disp.ax_.set_title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(f"results/figures/cm_{name}.png")
    plt.close()

# Model summary
results_df = pd.DataFrame(results)

# Plot model comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars="Model"), x="Model", y="value", hue="variable")
plt.title("Classification Model Performance and Training Time")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("results/figures/classification_model_comparison.png")
plt.show()

results_df, best_models["GradientBoosting"]
