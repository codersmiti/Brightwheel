from flask import Flask, request, jsonify, render_template
from triage import triage_message, calculate_weekly_impact
import pandas as pd
import os

app = Flask(__name__, static_folder='static')
def load_messages():
    try:
        df = pd.read_excel("data/AI Automation Builder Exercise CSV Data.xlsx")        
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        return []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/triage", methods=["POST"])
def triage():
    data = request.json
    message = data.get("message", "").strip()
    sender_name = data.get("sender_name", "there").strip()
    sender_email = data.get("sender_email", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    result = triage_message(message, sender_name, sender_email)
    result["weekly_impact"] = calculate_weekly_impact()
    return jsonify(result)

@app.route("/batch", methods=["GET"])
def batch():
    messages = load_messages()
    if not messages:
        return jsonify({"error": "Could not load CSV"}), 500

    results = []
    for msg in messages:
        body = str(msg.get("body", ""))
        sender = str(msg.get("sender_name", "there"))
        email = str(msg.get("sender_email", ""))
        subject = str(msg.get("subject", ""))
        full_message = f"Subject: {subject}\n\n{body}"

        result = triage_message(full_message, sender, email)
        result["message_id"] = msg.get("message_id")
        result["sender_name"] = sender
        result["subject"] = subject
        result["received_at"] = str(msg.get("received_at", ""))
        results.append(result)

    return jsonify(results)

@app.route("/debug")
def debug():
    try:
        df = pd.read_excel("data/AI Automation Builder Exercise CSV Data.xlsx")
        return jsonify({
            "columns": list(df.columns),
            "first_row": df.iloc[0].to_dict() if len(df) > 0 else {}
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/impact", methods=["GET"])
def impact():
    return jsonify(calculate_weekly_impact())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)