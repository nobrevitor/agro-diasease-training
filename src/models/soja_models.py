import torch
import torch.nn as nn
from torchvision import models


# MODELO CUSTOMIZADO (BASELINE)

class CNN_Simples(nn.Module):
    """
    CNN simples para servir como baseline no projeto de soja.

    Usa 4 blocos convolucionais com BatchNorm, ReLU e Pooling,
    finalizando com AdaptiveAvgPool para funcionar com imagens
    de diferentes tamanhos.
    """

    def __init__(self, num_classes):
        super().__init__()

        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.extrator_caracteristicas(x)
        x = self.classificador(x)
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


# DenseNet121
def get_densenet121(num_classes, freeze_backbone=True):
    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    if freeze_backbone:
        freeze_model(model)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


# FUNÇÃO CENTRAL PARA PIPELINE / MLFLOW

def get_model(model_name, num_classes, freeze_backbone=True):
    model_name = model_name.lower()

    if model_name in ["cnn_simples", "simplecnn"]:
        return CNN_Simples(num_classes)

    elif model_name == "resnet18":
        return get_resnet18(num_classes, freeze_backbone)

    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0(num_classes, freeze_backbone)

    elif model_name == "densenet121":
        return get_densenet121(num_classes, freeze_backbone)

    else:
        raise ValueError(f"Modelo '{model_name}' não suportado.")