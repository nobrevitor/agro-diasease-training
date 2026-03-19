import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from src.models.milho_models import get_model

def find_best_lr(
    model_name,
    num_classes,
    train_loader,
    device,
    freeze_backbone=True,
    start_lr=1e-7,
    end_lr=1,
    num_iter=100):


    model = get_model(model_name, num_classes, freeze_backbone)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    criterion = torch.nn.CrossEntropyLoss()

    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    lr_finder.range_test(
        train_loader,
        end_lr=end_lr,
        num_iter=num_iter
    )

    lrs = lr_finder.history["lr"]
    losses = lr_finder.history["loss"]

    losses_np = torch.tensor(losses)
    grads = torch.gradient(losses_np)[0]
    min_grad_idx = torch.argmin(grads).item()

    suggested_lr = lrs[min_grad_idx]

    print(f"LR sugerido: {suggested_lr:.6f}")

    fig = lr_finder.plot()

    lr_finder.reset()

    return suggested_lr, fig

    
def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / len(loader), correct / total


def run_training(
    model_name,
    num_classes,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    freeze_backbone,
    device,
    early_stopping=True,
    patience=5,
    use_onecycle=False
):

    model = get_model(model_name, num_classes, freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )
        early_stopping = False
        patience = None
    else:
        scheduler = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    counter = 0

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if early_stopping and not use_onecycle:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print("Early stopping acionado.")
                break

    if not use_onecycle:
        model.load_state_dict(best_model_wts)

    return model, history