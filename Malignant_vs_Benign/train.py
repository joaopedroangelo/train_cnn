import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np

# ========================
# Configurações Iniciais
# ========================
device = "cpu"
IMG_SIZE = 112
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-4

# ========================
# Definição do Modelo
# ========================
class CustomCNN(nn.Module):
    def __init__(self):
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========================
# Preprocessamento de Imagem
# ========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================
# Função para carregar DataLoaders
# ========================
def get_dataloaders(train_dir, val_dir, batch_size):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_data.class_to_idx

# ========================
# Função de Treinamento
# ========================
def train_model(model, train_loader, val_loader, epochs, optimizer, device):
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            weights = torch.where(labels == 0, 3.7, 1.0).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.BCEWithLogitsLoss(weight=weights)(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc = validate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Accuracy: {val_acc:.4f}")

# ========================
# Validação durante treino
# ========================
def validate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

# ========================
# Avaliação Final
# ========================
def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print("\n📊 Final Evaluation Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (Benign - 0): {precision[0]:.4f}")
    print(f"Precision (Malignant - 1): {precision[1]:.4f}")
    print(f"Recall (Benign - 0)   : {recall[0]:.4f}")
    print(f"Recall (Malignant - 1): {recall[1]:.4f}")
    print(f"F1 Score (Benign - 0) : {f1[0]:.4f}")
    print(f"F1 Score (Malignant - 1): {f1[1]:.4f}")

    return acc, precision, recall, f1

# ========================
# Predição de Imagem Única
# ========================
def predict_image(image_path, model, transform, device, threshold=0.5):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = int(prob > threshold)

    print(f"\nPrediction for '{image_path}':")
    print(f"Score: {prob:.4f} | Class: {'Malignant' if prediction else 'Benign'}")
    return prediction, prob

# ========================
# Execução Principal
# ========================
if __name__ == "__main__":
    train_loader, val_loader, class_map = get_dataloaders('./dataset/train', './dataset/test', BATCH_SIZE)
    print(class_map)

    model = CustomCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Treinamento
    train_model(model, train_loader, val_loader, EPOCHS, optimizer, device)
    torch.save(model.state_dict(), "model.pth")

    # Carregar modelo treinado
    model.load_state_dict(torch.load("model.pth", map_location=device))
    evaluate_model(model, val_loader, device)

    # Exemplo de predição
    # predict_image("./opa/5.jpeg", model, transform, device)
