import torch
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import traceback

from model import load_model
from inference import GradCAMPlusPlus, predict

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR.parent / 'models' / 'resnet50_rice.safetensors'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[app] Device: {device}')

model  = load_model(MODEL_PATH, device)
gradcam = GradCAMPlusPlus(model, device)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB limit

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG.'}), 400

    try:
        file_bytes = file.read()
        result     = predict(file_bytes, model, gradcam)
        return jsonify(result)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Internal error during prediction.'}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'device': str(device)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
