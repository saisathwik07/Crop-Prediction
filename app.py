from flask import Flask, render_template, request, jsonify
from core_app import SoilAnalysisCore
import base64

# GPT4All chatbot import (optional)
try:
    from gpt4all import GPT4All
    GPTALL_AVAILABLE = True
except ImportError:
    GPTALL_AVAILABLE = False
    GPT4All = None

import threading
gpt4all_lock = threading.Lock()
gpt_model = None
def get_gpt_model():
    global gpt_model
    if not GPTALL_AVAILABLE:
        return None
    if gpt_model is None:
        # Use the smallest model for low memory (update filename as needed)
        gpt_model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    return gpt_model


from tts_api import tts_api
app = Flask(__name__)
app.register_blueprint(tts_api)
core = SoilAnalysisCore()

# ================= PAGE ROUTES (GET) =================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/data")
def data_page():
    return render_template("data.html")

@app.route("/soil")
def soil_page():
    return render_template("soil.html")

@app.route("/prediction", methods=["GET"])
def prediction_page():
    return render_template("prediction.html")

@app.route("/iot")
def iot_page():
    return render_template("iot.html")

@app.route("/micronutrient", methods=["GET"])
def micronutrient_page():
    return render_template("micronutrient.html")

# ================= ACTION ROUTES (POST / API) =================

@app.route("/data/upload", methods=["POST"])
def upload_data():
    info = core.load_dataset(request.files['file'])
    return jsonify(info)

@app.route("/data/url", methods=["POST"])
def load_url():
    return jsonify(core.load_dataset_from_url(request.form['url']))

@app.route("/data/process", methods=["POST"])
def process_data():
    # Check if dataset attribute exists and is loaded
    if not hasattr(core, 'dataset') or core.dataset is None:
        return jsonify({"error": "Dataset not loaded. Please upload your dataset first."}), 400
    try:
        return jsonify(core.process_dataset())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analysis/train", methods=["POST"])
def train():
    return jsonify({"rmse": core.train_decision_tree()})

@app.route("/analysis/evaluate")
def evaluate():
    acc, report = core.evaluate_model()
    return jsonify({"accuracy": acc, "report": report})

@app.route("/prediction", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        payload = request.get_json()
        result = core.predict_crop(payload)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/iot/toggle", methods=["POST"])
def iot_toggle():
    return jsonify({"running": core.toggle_iot()})

@app.route("/iot/data")
def iot_data():
    # Fetch and store IoT data from Firebase, then return the latest data
    try:
        data = core.fetch_and_store_iot_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/micronutrient", methods=["POST"])
def micronutrient():
    return jsonify(core.analyze_micronutrients(request.json))
@app.route("/analysis/train-all", methods=["POST"])
def train_all_models():
    return jsonify(core.train_multiple_models())


@app.route("/analysis/soil", methods=["POST"])
def soil_analysis():
    data = request.json
    result = core.soil_health_analysis(data)
    graph = core.plot_micronutrient_graph(data)
    result["graph"] = graph
    return jsonify(result)


@app.route("/prediction/full", methods=["POST"])
def full_prediction():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON body"}), 400
        payload = request.get_json()
        if not payload or 'inputs' not in payload or 'micronutrients' not in payload:
            return jsonify({"error": "Missing 'inputs' or 'micronutrients' in payload"}), 400
        result = core.full_crop_recommendation(payload["inputs"], payload["micronutrients"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route("/model")
def model_page():
    return render_template("model.html")

# ================= RUN =================

# ================= CHATBOT ROUTE =================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    if not GPTALL_AVAILABLE:
        return jsonify({"response": "Chatbot feature not available. Install gpt4all: pip install gpt4all"}), 503
    
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    try:
        with gpt4all_lock:
            model = get_gpt_model()
            with model.chat_session():
                response = model.generate(user_message, max_tokens=128, temp=0.7)
        return jsonify({"response": response.strip()}), 200
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)