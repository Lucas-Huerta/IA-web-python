"""
Script d'entraînement MNIST - Adapté du PyTorch Quickstart
Classification de chiffres manuscrits (0-9) avec export ONNX pour le web
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import os

print("=== Classification MNIST avec CNN - PyTorch ===")

######################################################################
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

######################################################################
# données en batches

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# device - Configuration optimisée pour MacBook
device = (
    "mps" 
    if torch.backends.mps.is_available() 
    else "cpu"
)
print(f"Using {device} device")

if device == "mps":
    print("MPS (Metal Performance Shaders) détecté - Optimal pour MacBook Apple Silicon!")
else:
    print(" Utilisation du CPU - MPS non disponible")

# Define CNN model - Architecture convolutionnelle simple et efficace
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Première convolutionnelle
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Deuxième  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Troisième 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # → 32x28x28
        x = self.pool(x)                 # → 32x14x14
        
        x = F.relu(self.conv2(x))        # → 64x14x14
        x = self.pool(x)                 # → 64x7x7
        
        x = F.relu(self.conv3(x))        # → 128x7x7
        x = self.pool(x)                 # → 128x3x3
        x = self.dropout1(x)
        
        # Aplatissement
        x = torch.flatten(x, 1)          # 128x3x3 → 1152
        
        x = F.relu(self.fc1(x))          # 1152 → 256
        x = self.dropout2(x)
        x = self.fc2(x)                  # 256 → 10
        
        return x

model = DigitCNN().to(device)
print(model)

# Compter les paramètres
total_params = sum(p.numel() for p in model.parameters())
print(f"Nombre total de paramètres: {total_params:,}")

######################################################################
# Optimizing the Model Parameters
# --------------------------------

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam au lieu de SGD

######################################################################
# Training and Testing Functions
# -------------------------------

def train(dataloader, model, loss_fn, optimizer):
    """Entraîne le modèle sur une époque"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    """Teste le modèle et retourne la précision"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy

######################################################################
# Training Loop
# -------------

epochs = 5
best_accuracy = 0

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = test(test_dataloader, model, loss_fn)
    
    # Sauvegarder le meilleur modèle
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_digit_cnn.pth")
        print(f" Meilleur modèle CNN sauvegardé (précision: {accuracy:.1f}%)")

print("Done!")
print(f"Meilleure précision CNN: {best_accuracy:.1f}%")

######################################################################
# Saving Models
# -------------

print("Sauvegarde du modèle CNN final...")
torch.save(model.state_dict(), "digit_cnn.pth")
print("Saved CNN Model State to digit_cnn.pth")

######################################################################
# Loading Models for ONNX Export

# Charger le meilleur modèle CNN
model = DigitCNN().to(device)
model.load_state_dict(torch.load("best_digit_cnn.pth", weights_only=True))
print("Modèle CNN rechargé pour l'export ONNX")

######################################################################
# Export to ONNX for Web

def export_to_onnx(model, model_path="web/digit_cnn.onnx"):
    """Exporte le modèle CNN vers ONNX pour l'inférence web"""
    
    # Créer le dossier web
    os.makedirs('web', exist_ok=True)
    
    model.eval()
    
    # Exemple d'entrée pour l'export (image 1x1x28x28)
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    print(f"modèle CNN sur {model_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Modèle CNN exporté vers {model_path}")
export_to_onnx(model)

# Classes des chiffres (0-9)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model.eval()
print("\nTest du modèle CNN:")

for i in range(5):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x.unsqueeze(0)) 
        predicted_class = pred[0].argmax(0).item()
        confidence = torch.softmax(pred[0], dim=0)[predicted_class] * 100
        
        print(f"Échantillon {i+1}: Prédit={predicted_class}, "
              f"Réel={y}, Confiance={confidence:.1f}%")
print(f"Architecture: CNN avec {total_params:,} paramètres")
print("Fichiers créés:")
print("   - best_digit_cnn.pth (meilleur modèle CNN)")
print("   - digit_cnn.pth (modèle CNN final)")
print("   - web/digit_cnn.onnx (modèle CNN pour le web)")
print("\n Lancez le serveur web: cd web && python -m http.server 8000")