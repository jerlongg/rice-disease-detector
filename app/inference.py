import cv2
import numpy as np
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from PIL import Image

from model import CLASSES, IDX_TO_CLASS, DISEASE_INFO

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOW_CONFIDENCE_THRESHOLD = 0.50


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return (img - MEAN) / STD


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())


def read_image(file_bytes: bytes):
    """Read image bytes → RGB numpy array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('Could not decode image. Please upload a valid JPG or PNG.')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Hooks the final feature layer for both ResNet-50 and MobileNetV3.
    ResNet-50: layer4 (index 7 in backbone children)
    MobileNetV3: features block (features[-1])
    """
    def __init__(self, model, device, model_type='resnet50'):
        self.model       = model
        self.device      = device
        self.model_type  = model_type
        self.activations = None
        self.gradients   = None
        
        if model_type == 'mobilenetv3':
            target = model.features[-1]  # Last features block
        else:  # resnet50
            target = list(model.backbone.children())[7]  # layer4
        
        target.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach()))
        target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))

    def generate(self, tensor: torch.Tensor):
        self.model.eval()
        tensor = tensor.unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
        pred   = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, pred].backward()

        weights = (self.gradients ** 2).mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1)).squeeze()
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, pred, probs.cpu().detach().numpy()


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr     = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def img_to_b64(rgb: np.ndarray) -> str:
    pil = Image.fromarray(rgb.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Main predict function ─────────────────────────────────────────────────────

def predict(file_bytes: bytes, model, gradcam, model_name: str = 'resnet50') -> dict:
    img_rgb = read_image(file_bytes)
    tensor  = to_tensor(preprocess(img_rgb))

    cam, pred_idx, probs = gradcam.generate(tensor)

    confidence  = float(probs[pred_idx])
    class_name  = IDX_TO_CLASS[pred_idx]
    info        = DISEASE_INFO[class_name]
    low_conf    = confidence < LOW_CONFIDENCE_THRESHOLD

    # Build overlay
    overlay_rgb = overlay_heatmap(img_rgb, cam)

    # Resize original for display
    orig_disp = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    all_scores = [
        {'class': CLASSES[i], 'label': DISEASE_INFO[CLASSES[i]]['label'],
         'score': float(probs[i])}
        for i in range(len(CLASSES))
    ]
    all_scores.sort(key=lambda x: x['score'], reverse=True)

    return {
        'model':       model_name,
        'class':       class_name,
        'label':       info['label'],
        'type':        info['type'],
        'confidence':  round(confidence * 100, 2),
        'description': info['description'],
        'action':      info['action'],
        'severity':    info['severity'],
        'low_conf':    low_conf,
        'original_b64': img_to_b64(orig_disp),
        'heatmap_b64':  img_to_b64(overlay_rgb),
        'all_scores':   all_scores,
    }
