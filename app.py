from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    plot_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Load dataset
            data = pd.read_csv(filepath)

            X_raw = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Convert categorical features to numeric using one-hot encoding
            X = pd.get_dummies(X_raw, drop_first=True)

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scaling
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC()
            }

            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
                rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                results[name] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "confusion_matrix": cm.tolist()
                }

            # Plot Accuracy Graph
            names = list(results.keys())
            accuracies = [results[m]["accuracy"] for m in names]

            plt.figure()
            plt.bar(names, accuracies)
            plt.title("Model Accuracy Comparison")
            plt.xticks(rotation=30)

            plot_path = os.path.join("static", "accuracy.png")
            os.makedirs("static", exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

    return render_template("index.html", results=results, plot_path=plot_path)


if __name__ == "__main__":
    app.run(debug=True)