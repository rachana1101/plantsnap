import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import argparse

# Add at top after imports:
parser = argparse.ArgumentParser()
parser.add_argument('--data', 
                    default='plantsnap/herbs/train',
                    help='Path to training data')
args = parser.parse_args()

# Then use args.data instead of hardcoded path:
train_dataset = datasets.ImageFolder(args.data,
                                     transform=transform)

# ── MLflow Setup ─────────────────────────────────────
# Tracks every training run automatically
# View results: mlflow ui → http://localhost:5000
mlflow.set_experiment("PlantSnap-Herb-Classifier")

# ── Hyperparameters ───────────────────────────────────
LR         = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 10
# NUM_CLASSES set dynamically from dataset below ✅
# Add new herb folder → automatically picked up!

# ── Image Transforms ──────────────────────────────────
# Resize(256) first → CenterCrop(224) to avoid losing content from small images
# Normalize values match ImageNet statistics ResNet18 was trained on
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset + DataLoader ──────────────────────────────
# ImageFolder reads folder names as labels automatically
# shuffle=True prevents ordered bias across epochs
train_dataset = datasets.ImageFolder('plantsnap/herbs/train',
                                     transform=transform)
train_loader  = DataLoader(train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True)

# Dynamic class count — just add a new herb folder and retrain! ✅
NUM_CLASSES = len(train_dataset.classes)
print(f"✅ {len(train_dataset)} images | {NUM_CLASSES} classes")

# ── Device ────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# ── Model ─────────────────────────────────────────────
# Freeze backbone — preserves ImageNet knowledge (edges, textures, shapes)
# Only train final layer (512 → 70) — just 51K of 11M parameters
model = models.resnet18(weights='IMAGENET1K_V1')

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, NUM_CLASSES)
model    = model.to(device)

# ── Loss + Optimizer + Scheduler ─────────────────────
criterion = nn.CrossEntropyLoss()

# Only pass fc.parameters() — no point optimizing frozen layers
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

# Halves lr when loss plateaus for 3 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ── Training Loop with MLflow ─────────────────────────
print("🚀 Starting training...")

with mlflow.start_run():

    # Log all hyperparameters at once
    mlflow.log_params({
        "learning_rate": LR,
        "batch_size":    BATCH_SIZE,
        "epochs":        NUM_EPOCHS,
        "num_classes":   NUM_CLASSES,
        "model":         "resnet18",
        "optimizer":     "adam",
        "frozen_layers": True,
        "device":        str(device)
    })

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        correct      = 0
        total        = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Track accuracy
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Epoch metrics
        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        # Learning rate tracking
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        new_lr = optimizer.param_groups[0]['lr']

        lr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {accuracy:.1f}%{lr_note}")

        # Log metrics to MLflow — one line per metric per epoch
        mlflow.log_metrics({
            "train_loss":     avg_loss,
            "train_accuracy": accuracy,
            "learning_rate":  new_lr
        }, step=epoch)

    # Save model as MLflow artifact
    torch.save(model.state_dict(), 'my_herbs_model.pth')
    mlflow.log_artifact('my_herbs_model.pth')

    # Log final summary metrics
    mlflow.log_metrics({
        "final_loss":     avg_loss,
        "final_accuracy": accuracy
    })

    print(f"\n✅ Training complete!")
    print(f"   Final loss:     {avg_loss:.4f}")
    print(f"   Final accuracy: {accuracy:.1f}%")
    print(f"   Run logged to MLflow ✅")
    print(f"   View: mlflow ui → http://localhost:5000")

print("💾 SAVED: my_herbs_model.pth")
print("🎉 70-HERB CLASSIFIER READY!")