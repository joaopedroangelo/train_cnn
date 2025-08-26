import torch
import torch.nn as nn
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

# =============================
# Configurações
# =============================
device = "cpu"
img_size = 256

# =============================
# Transforms
# =============================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# Caminhos corretos
# =============================
train_path = './dataset/train'
val_path = './dataset/val'
test_path = './dataset/test'

# =============================
# Carregamento dos datasets
# =============================
train_data = datasets.ImageFolder(train_path, transform=transform)
val_data = datasets.ImageFolder(val_path, transform=transform)
test_data = datasets.ImageFolder(test_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

num_classes = len(train_data.classes)
print(f"📌 Número de classes: {num_classes} -> {train_data.classes}")

# =============================
# Pesos para balanceamento
# =============================
class_weights = compute_class_weight(
    'balanced', classes=np.arange(num_classes), y=train_data.targets
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# =============================
# Modelo CNN
# =============================
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        final_size = img_size // (2**5)  # 256 -> 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * final_size * final_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# =============================
# Inicialização
# =============================
model = CustomCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =============================
# Função de Treino
# =============================
def train_model(model, train_loader, val_loader, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validação
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Accuracy: {acc:.4f}")

# =============================
# Avaliação Final
# =============================
def evaluate_model(model, loader, dataset_name="Val"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n📊 Métricas ({dataset_name}):")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nDetalhado por classe:")
    print(classification_report(y_true, y_pred, target_names=train_data.classes))

    return acc, precision, recall, f1

# =============================
# Predição para uma Imagem
# =============================
def predict_image(image_path, transform):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    print(f"\n🖼️ Predição: {train_data.classes[prediction]} ({probs[0][prediction]:.4f})")
    return prediction, probs[0][prediction].item()

# =============================
# Execução Principal
# =============================
if __name__ == "__main__":
    print(train_data.class_to_idx)
    train_model(model, train_loader, val_loader, epochs=100)
    torch.save(model.state_dict(), "model_seven_class.pth")

    # Carrega modelo salvo
    model.load_state_dict(torch.load("model_seven_class.pth", map_location=device))

    # Avaliação em validação e teste
    evaluate_model(model, val_loader, dataset_name="Validação")
    evaluate_model(model, test_loader, dataset_name="Teste Final")

    # Exemplo de predição
    # predict_image("./dataset/test/akiec/ISIC_0024339.jpg", transform)
