from flask import Flask, render_template, request
import pandas as pd
import os
import sqlite3
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# DATABASE CONNECTION
# ----------------------------
def get_db_connection():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# ----------------------------
# CREATE TABLES
# ----------------------------
def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        upload_time TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        upload_id INTEGER,
        model_name TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        confusion_matrix TEXT,
        FOREIGN KEY(upload_id) REFERENCES uploads(id)
    )
    """)

    conn.commit()
    conn.close()

create_tables()

# ----------------------------
# MAIN ROUTE
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Save upload info
            conn = get_db_connection()
            cur = conn.cursor()

            cur.execute(
                "INSERT INTO uploads (filename, upload_time) VALUES (?, ?)",
                (file.filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            upload_id = cur.lastrowid

            # Load dataset
            data = pd.read_csv(filepath)

            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale
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
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                cm = confusion_matrix(y_test, y_pred)

                # Save results to DB
                cur.execute("""
                    INSERT INTO results 
                    (upload_id, model_name, accuracy, precision, recall, confusion_matrix)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    upload_id,
                    name,
                    acc,
                    prec,
                    rec,
                    str(cm.tolist())
                ))

                results[name] = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "confusion_matrix": cm.tolist()
                }

            conn.commit()
            conn.close()

    return render_template("index.html", results=results)

# ----------------------------
# VIEW HISTORY
# ----------------------------
@app.route("/history")
def history():
    conn = get_db_connection()
    uploads = conn.execute("SELECT * FROM uploads").fetchall()

    data = []
    for upload in uploads:
        results = conn.execute(
            "SELECT * FROM results WHERE upload_id=?",
            (upload["id"],)
        ).fetchall()

        data.append({
            "file": upload["filename"],
            "time": upload["upload_time"],
            "results": results
        })

    conn.close()
    return render_template("history.html", data=data)

# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)