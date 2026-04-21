import torch
import torch.nn as nn
from torchvision import models
from safetensors.torch import load_file
from pathlib import Path

CLASSES = ['Bacterial_Blight', 'Blast', 'Brown_Spot', 'Tungro', 'Healthy_Rice_Leaf', 'Hispa']
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

DISEASE_INFO = {
    'Bacterial_Blight': {
        'label': 'Bacterial Blight',
        'type': 'Bacterial',
        'description': 'Water-soaked to yellowish stripes on leaf margins, eventually turning white to gray.',
        'action': 'Use copper-based bactericides. Remove infected plants. Avoid flooding during early growth.',
        'severity': 'high',
    },
    'Blast': {
        'label': 'Leaf Blast',
        'type': 'Fungal',
        'description': 'Diamond-shaped lesions with gray centers and brown borders on leaves.',
        'action': 'Apply fungicides (tricyclazole). Avoid excessive nitrogen. Ensure proper field drainage.',
        'severity': 'high',
    },
    'Brown_Spot': {
        'label': 'Brown Spot',
        'type': 'Fungal',
        'description': 'Circular to oval brown spots with yellow halos scattered across the leaf.',
        'action': 'Apply mancozeb or iprodione fungicides. Improve soil nutrition, especially potassium.',
        'severity': 'medium',
    },
    'Tungro': {
        'label': 'Tungro Virus',
        'type': 'Viral',
        'description': 'Yellow-orange discoloration of leaves, stunted growth, and reduced tillering.',
        'action': 'No cure. Remove and destroy infected plants. Control leafhopper vectors with insecticides.',
        'severity': 'high',
    },
    'Healthy_Rice_Leaf': {
        'label': 'Healthy Leaf',
        'type': 'None',
        'description': 'The leaf appears healthy with no visible signs of disease.',
        'action': 'Continue regular monitoring and good agricultural practices.',
        'severity': 'none',
    },
    'Hispa': {
        'label': 'Rice Hispa',
        'type': 'Insect',
        'description': 'White streaks parallel to the leaf veins caused by larvae scraping the leaf surface.',
        'action': 'Apply chlorpyrifos or quinalphos. Clip and destroy affected leaf tips. Avoid dense planting.',
        'severity': 'medium',
    },
}


class RiceResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(model_path: str, device: torch.device) -> RiceResNet:
    model = RiceResNet(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(load_file(str(model_path)))
    model.eval()
    print(f'[model] Loaded from {model_path}  ({NUM_CLASSES} classes)')
    return model
