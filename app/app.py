import torch
from flask import Flask, request, jsonify, render_template
from pathlib import Path
import traceback

from model import load_model
from inference import GradCAMPlusPlus, predict

# ── Setup ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR.parent / 'models'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[app] Device: {device}')

# Load both models
models_dict = {
    'resnet50': load_model(MODELS_DIR / 'resnet50_rice.safetensors', device, model_type='resnet50'),
    'mobilenetv3': load_model(MODELS_DIR / 'mobilenetv3_rice.safetensors', device, model_type='mobilenetv3'),
}
gradcams = {
    'resnet50': GradCAMPlusPlus(models_dict['resnet50'], device, model_type='resnet50'),
    'mobilenetv3': GradCAMPlusPlus(models_dict['mobilenetv3'], device, model_type='mobilenetv3'),
}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB limit

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
AVAILABLE_MODELS = ['resnet50', 'mobilenetv3']
DEFAULT_MODEL = 'resnet50'

def allowed(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/models', methods=['GET'])
def get_models():
    """Return list of available models."""
    return jsonify({'models': AVAILABLE_MODELS, 'default': DEFAULT_MODEL})


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400
    if not allowed(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload JPG or PNG.'}), 400

    # Get selected model, default to resnet50
    model_name = request.form.get('model', DEFAULT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model. Available: {AVAILABLE_MODELS}'}), 400

    try:
        file_bytes = file.read()
        model = models_dict[model_name]
        gradcam = gradcams[model_name]
        result = predict(file_bytes, model, gradcam, model_name=model_name)
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
