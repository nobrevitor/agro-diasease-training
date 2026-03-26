import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# FUNÇÃO BASE: GERA PREDIÇÕES
# --------------------------------------------------
def get_predictions(model, loader, device):

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# --------------------------------------------------
# FUNÇÃO DE AVALIAÇÃO COMPLETA
# --------------------------------------------------
def evaluate_model(model, loader, device, class_names=None):

    y_true, y_pred = get_predictions(model, loader, device)

    acc = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }