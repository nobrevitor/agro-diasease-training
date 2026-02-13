import torch
import torch.nn as nn
from torchvision import models


# MODELO CUSTOMIZADO (BASELINE)
class SimpleCNN(nn.Module):
    """
    CNN simples para servir como baseline.
    Usa AdaptiveAvgPool para funcionar com qualquer
    imagem >= 64x64.
    """

    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# MODELOS PRETRAINED (TRANSFER LEARNING)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


# ResNet18
def get_resnet18(num_classes, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        freeze_model(model)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# EfficientNet-B0
def get_efficientnet_b0(num_classes, freeze_backbone=True):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    if freeze_backbone:
        freeze_model(model)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# -------- DenseNet121 --------
def get_densenet121(num_classes, freeze_backbone=True):
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    if freeze_backbone:
        freeze_model(model)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


# =====================================================
# 3️⃣ FUNÇÃO CENTRAL PARA PIPELINE / MLFLOW
# =====================================================

def get_model(model_name, num_classes, freeze_backbone=True):
    model_name = model_name.lower()

    if model_name == "simplecnn":
        return SimpleCNN(num_classes)

    elif model_name == "resnet18":
        return get_resnet18(num_classes, freeze_backbone)

    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, freeze_backbone)

    elif model_name == "densenet121":
        return get_densenet121(num_classes, freeze_backbone)

    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")
