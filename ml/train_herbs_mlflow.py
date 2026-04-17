import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import argparse

# ── Args ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data',
                    help='Path to data folder containing train/ and val/')
args = parser.parse_args()

# ── MLflow Setup ──────────────────────────────────────
mlflow.set_experiment("PlantSnap-Herb-Classifier")

# ── Hyperparameters ───────────────────────────────────
LR         = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 60    # more epochs for fine-tuning

# ── Image Transforms ──────────────────────────────────
# Train: augmentation helps model generalise to new herb photos
# Val:   no augmentation — measure real performance on clean images
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Dataset + DataLoader ──────────────────────────────
train_dataset = datasets.ImageFolder(f"{args.data}/train",
                                     transform=train_transform)
val_dataset   = datasets.ImageFolder(f"{args.data}/val",
                                     transform=val_transform)

train_loader  = DataLoader(train_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=True,
                           num_workers=0)   # 0 = safe for MPS on Mac
val_loader    = DataLoader(val_dataset,
                           batch_size=BATCH_SIZE,
                           shuffle=False,
                           num_workers=0)

NUM_CLASSES = len(train_dataset.classes)
print(f"✅ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {NUM_CLASSES}")

# ── Device ────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

# ── Model ─────────────────────────────────────────────
model = models.resnet18(weights='IMAGENET1K_V1')

# Step 1: freeze ALL layers
for param in model.parameters():
    param.requires_grad = False

# Step 2: unfreeze layer4 — learns herb-specific features ← KEY CHANGE!
for param in model.layer4.parameters():
    param.requires_grad = True

# Step 3: replace classifier for our herb classes
model.fc = nn.Linear(512, NUM_CLASSES)
# fc is always trainable (new layer = random weights)

model = model.to(device)

# Count trainable params for logging
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"🔓 Trainable params: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.1f}%)")

# ── Loss + Optimizer + Scheduler ─────────────────────
criterion = nn.CrossEntropyLoss()

# Discriminative learning rates:
#   layer4 = fine-tune gently (10x smaller LR)  ← KEY CHANGE!
#   fc     = learn fast (full LR — new random weights)
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': LR * 0.1},
    {'params': model.fc.parameters(),     'lr': LR}
])

# patience=4 now that two layer groups are updating
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=4
)

# ── Validation Function ───────────────────────────────
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct  = 0
    total    = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            val_loss    += loss.item()
            _, predicted = outputs.max(1)
            total       += labels.size(0)
            correct     += predicted.eq(labels).sum().item()

    return val_loss / len(val_loader), 100.0 * correct / total

# ── Training Loop ─────────────────────────────────────
print("🚀 Starting Run 4 — layer4 unfrozen!")
print(f"   layer4 LR: {LR * 0.1:.5f}  |  fc LR: {LR:.5f}")
print()

best_val_acc = 0.0

with mlflow.start_run():

    mlflow.log_params({
        "learning_rate":      LR,
        "layer4_lr":          LR * 0.1,
        "batch_size":         BATCH_SIZE,
        "epochs":             NUM_EPOCHS,
        "num_classes":        NUM_CLASSES,
        "model":              "resnet18",
        "optimizer":          "adam",
        "frozen_layers":      "layer1,layer2,layer3",
        "unfrozen_layers":    "layer4,fc",
        "augmentation":       True,
        "scheduler_patience": 4,
        "device":             str(device),
        "trainable_params":   trainable,
        "run":                4
    })

    for epoch in range(NUM_EPOCHS):

        # ── Train ─────────────────────────────────────
        model.train()
        running_loss  = 0
        train_correct = 0
        train_total   = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            _, predicted   = outputs.max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        # ── Validate ──────────────────────────────────
        avg_val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )

        # ── Save best model ───────────────────────────
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'my_herbs_best.pth')
            best_note = f"  💾 Best! ({val_accuracy:.1f}%)"
        else:
            best_note = ""

        # ── Learning Rate ─────────────────────────────
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        lr_note = f"  📉 LR→{new_lr:.6f}" if new_lr < old_lr else ""

        # ── Print ─────────────────────────────────────
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.1f}%"
              f"{lr_note}{best_note}")

        # ── MLflow ────────────────────────────────────
        mlflow.log_metrics({
            "train_loss":     avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss":       avg_val_loss,
            "val_accuracy":   val_accuracy,
            "learning_rate":  new_lr,
            "best_val_acc":   best_val_acc
        }, step=epoch)

    # ── Save final model ──────────────────────────────
    torch.save(model.state_dict(), 'my_herbs_model.pth')
    mlflow.log_artifact('my_herbs_model.pth')
    mlflow.log_artifact('my_herbs_best.pth')

    mlflow.log_metrics({
        "final_train_loss": avg_train_loss,
        "final_val_loss":   avg_val_loss,
        "final_val_acc":    val_accuracy,
        "best_val_acc":     best_val_acc
    })

    print(f"\n✅ Training complete!")
    print(f"   Final  → Train: {train_accuracy:.1f}% | Val: {val_accuracy:.1f}%")
    print(f"   Best   → Val Acc: {best_val_acc:.1f}%")
    print(f"   MLflow → mlflow ui → http://localhost:5000")

print(f"\n💾 SAVED:")
print(f"   my_herbs_best.pth  ← USE THIS for CoreML! ({best_val_acc:.1f}%)")
print(f"   my_herbs_model.pth ← final epoch")
print(f"\n🎉 Run 4 complete! layer4 unfrozen!")