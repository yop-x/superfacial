from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_DIR = BASE_DIR / "labels"
LABELS_DIR.mkdir(exist_ok=True)

CSV_PATH = LABELS_DIR / "labels.csv"

app = Flask(__name__)
CORS(app)  # allow requests from http://localhost:5500 etc.

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/save")
def save():
    """
    Expects JSON like:
    [
      {"name": "Adele", "label": "yes"},
      {"name": "Aaron Donald", "label": "no"},
      ...
    ]
    """
    data = request.get_json(silent=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON list of {name,label} objects"}), 400

    # validate rows
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        label = item.get("label")
        if not name or label not in ("yes", "no"):
            continue
        rows.append({"name": str(name), "label": label})

    if not rows:
        return jsonify({"error": "No valid rows found (need name + label yes/no)"}), 400

    df = pd.DataFrame(rows).drop_duplicates(subset=["name"], keep="last")
    df["saved_at"] = datetime.now().isoformat(timespec="seconds")

    df.to_csv(CSV_PATH, index=False)

    return jsonify({"status": "saved", "count": len(df), "path": str(CSV_PATH)}), 200


if __name__ == "__main__":
    # listen on localhost:8000
    app.run(host="127.0.0.1", port=8000, debug=True)
