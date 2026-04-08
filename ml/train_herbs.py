# ***

# ## What You Did: PlantSnap Training Summary

# ### The Project
# You trained a **ResNet18 model** on **3,461 herb images across 70 classes** using PyTorch, then iteratively improved training by tuning hyperparameters.

# ***

# ### Problem 1: NumPy Compatibility Error
# **What happened:** Training crashed with `RuntimeError: Numpy is not available`.

# **Root cause:** NumPy 2.0 changed its C API, breaking binary compatibility with PyTorch which was compiled against NumPy 1.x.

# **Fix:**
# ```bash
# pip install "numpy<2"
# ```

# **Interview answer:**
# > "I hit a dependency conflict between NumPy 2.0 and PyTorch. NumPy 2.0 broke the binary API PyTorch relied on. I downgraded to NumPy 1.26.4 which resolved it."

# ***

# ### Problem 2: Loss Dropping Too Slowly

# **What you observed across 24 epochs:**

# | Epoch | Loss | Key Observation |
# |-------|------|----------------|
# | 1 | 4.6567 | Near random baseline (ln 70 = 4.25) |
# | 10 | 4.2033 | Just crossed random baseline |
# | 24 | 3.9834 | Steady but very slow |

# **Root cause:** Learning rate was too low → model taking tiny steps each epoch (~0.013 drop/epoch).

# **Fix 1 — Increase epochs to 50:**
# ```python
# num_epochs = 50
# ```

# **Fix 2 — Add ReduceLROnPlateau scheduler:**
# ```python
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=3, verbose=True
# )
# scheduler.step(epoch_loss)  # at end of each epoch
# ```

# **Why this helps:**
# - The scheduler **automatically halves the learning rate** when loss stops improving for 3 consecutive epochs.
# - This allows the model to take big steps early (fast learning) and small steps later (fine-tuning), instead of tiny steps throughout.

# ***

# ### Interview Explanation (Deliver This)

# > "I trained ResNet18 on 3,461 herb images across 70 classes. Initially, loss was near the random baseline of ln(70)=4.25, which means the model was essentially guessing. After 24 epochs, loss dropped to 3.98, but the pace was too slow — about 0.013 per epoch. At that rate I'd need 100+ epochs to reach meaningful accuracy.
# >
# > The fix had two parts: I increased epochs to 50 and added a ReduceLROnPlateau scheduler. The scheduler monitored loss and halved the learning rate whenever improvement stalled for 3 consecutive epochs. This is important because a fixed learning rate is too aggressive for fine-tuning but too slow early on. The adaptive scheduler handles both phases automatically.
# >
# > I also hit a NumPy 2.0 compatibility issue with PyTorch mid-training, which I resolved by pinning NumPy to 1.26.4."

# ***

# ### Key Concepts to Know for Follow-up Questions

# **"Why not just use a very high learning rate from the start?"**
# > "High LR causes loss to oscillate or diverge. We saw minor upticks at epochs 10–11 and 17–18 even at moderate LR. The scheduler gives you the best of both worlds."

# **"What is an epoch?"**
# > "One complete pass through all 3,461 training images. Each epoch the model sees every herb once, updates weights once per batch, and ideally gets a bit better."

# **"Why ln(70) as the random baseline?"**
# > "With 70 equally likely classes, a random model assigns 1/70 probability to each. Cross-entropy of uniform random guessing = −log(1/70) = ln(70) ≈ 4.25."

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('plantsnap/herbs/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # no custom_collate

print(f"✅ Found {len(train_dataset)} images across {len(train_dataset.classes)} classes")
print(f"Your classes: {train_dataset.classes}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet18(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False                                  # freeze backbone
model.fc = nn.Linear(512, len(train_dataset.classes))            # trainable head only
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

NUM_EPOCHS = 10
print("🚀 Training YOUR herbs...")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)   # ✅ compute once, use once
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_loss)                      # ✅ called once
    new_lr = optimizer.param_groups[0]['lr']
    lr_note = f"  📉 LR → {new_lr:.6f}" if new_lr < old_lr else ""
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}{lr_note}")

torch.save(model.state_dict(), 'my_herbs_model.pth')
print("💾 SAVED: my_herbs_model.pth")
print("🎉 YOUR 4-HERB CLASSIFIER READY!")
