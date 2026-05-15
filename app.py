from flask import Flask, request, jsonify, render_template
from triage import triage_message, calculate_weekly_impact, detect_followup, SLA_BY_PRIORITY
import pandas as pd
import os
import time

app = Flask(__name__, static_folder='static')

def load_messages():
    try:
        df = pd.read_excel("data/AI Automation Builder Exercise CSV Data.xlsx")
        df = df.fillna("")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Could not load messages: {e}")
        return []

# Load once at startup — used for follow-up detection
MESSAGES = load_messages()

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

    # Follow-up detection — checks sender email against loaded dataset
    if sender_email:
        followup = detect_followup(message, sender_email, MESSAGES)
        if followup:
            result["is_followup"] = True
            result["original_message_id"] = followup["original_message_id"]
            result["original_subject"] = followup["original_subject"]
            
            # Find the original message and carry forward its category and owner
            original = next((m for m in MESSAGES if str(m.get("message_id", "")) == followup["original_message_id"]), None)
            
            if original:
                # Re-triage the original message to get its category and owner
                original_body = str(original.get("body", ""))
                original_subject = str(original.get("subject", ""))
                original_result = triage_message(f"Subject: {original_subject}\n\n{original_body}", sender_name)
                result["category"] = original_result["category"]
                result["owner"] = original_result["owner"]
                result["supporting_team"] = original_result["supporting_team"]
            
            # Always bump priority for follow-ups that haven't been responded to
            if result.get("priority") not in ("Critical", "High"):
                result["priority"] = "High"
            result["sla"] = SLA_BY_PRIORITY["High"]
        else:
            result["is_followup"] = False
    else:
        result["is_followup"] = False

    result["weekly_impact"] = calculate_weekly_impact()
    return jsonify(result)
@app.route("/batch", methods=["GET"])
def batch():
    try:
        messages = load_messages()
        if not messages:
            return jsonify({"error": "Could not load Excel — file may be missing or wrong name"}), 500

        results = []
        for i, msg in enumerate(messages):
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

            time.sleep(3)

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/impact", methods=["GET"])
def impact():
    return jsonify(calculate_weekly_impact())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)