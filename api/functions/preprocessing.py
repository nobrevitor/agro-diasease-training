from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    # Apenas redimensiona para 224x224
    transforms.Resize((224, 224)),

    # Converte para Tensor
    transforms.ToTensor(),

    # Normaliza da mesma forma que no treino
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_bytes):
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)