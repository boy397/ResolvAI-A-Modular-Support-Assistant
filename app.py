"""Flask serving app for Render. Loads artifacts once at startup (SupportAgent
is a lazy singleton) and exposes /predict for the agent and / for a minimal
manual test page."""
import os
from flask import Flask, request, jsonify, render_template

from resolvai import logger
from resolvai.pipeline.predict_pipeline import SupportAgent

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    customer_text = payload.get("message", "").strip()
    if not customer_text:
        return jsonify({"error": "missing 'message' field"}), 400

    try:
        agent = SupportAgent.instance()
        result = agent.handle(customer_text)
        return jsonify(result)
    except FileNotFoundError as e:
        logger.exception(e)
        return jsonify({"error": "artifacts not found -- run `python main.py` first to train/build them"}), 500
    except Exception as e:
        logger.exception(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
