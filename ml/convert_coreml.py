# =============================================================================
# PLANTSNAP HERB CLASSIFIER - ANNOTATED COREML CONVERSION CODE
# PyTorch → CoreML → iOS Deployment
# =============================================================================
#
# WHAT THIS FILE DOES
# ────────────────────
# Takes the trained model saved as my_herbs_model.pth and converts it
# into my_herbs.mlpackage — Apple's CoreML format — ready to drag into Xcode.
#
# WHY A SEPARATE CONVERSION SCRIPT?
# ───────────────────────────────────
# Training script ends → model object disappears from memory
# Only my_herbs_model.pth survives on disk
# This script rebuilds the model structure, loads the weights,
# wraps it for iOS, and exports it.
#
# THE COMPLETE PIPELINE:
# ──────────────────────
#   my_herbs_model.pth           ← saved weights from training
#           ↓
#   backbone = resnet18()        ← rebuild empty shell
#   load_state_dict()            ← fill with trained herb weights
#           ↓
#   NormalizedResNet(backbone)   ← wrap: bake in normalization + softmax
#   model.eval()                 ← switch to consistent prediction mode
#           ↓
#   torch.jit.trace()            ← record all operations as static graph
#           ↓
#   ct.convert()                 ← translate graph to CoreML format
#           ↓
#   my_herbs.mlpackage           ← drag into Xcode → works immediately! ✅
#
# ALSO IN THIS REPO: export_onnx.py
# ────────────────────────────────────
# An alternative export path: PyTorch → ONNX → CoreML
# ONNX = Open Neural Network Exchange (universal format — runs on Android,
# web, Raspberry Pi, Windows too). Not needed for iOS but useful to know.
# NOTE: That file has a bug — model.fc = Linear(512, 4) should be Linear(512, 70)!
import coremltools as ct
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# =============================================================================
# CLASS LABELS — Your 70 Herb Classes
# =============================================================================
# This list maps model output indices to herb names.
# Order MUST match exactly how ImageFolder sorted your training folders
# (alphabetical order). Index 0 = first folder alphabetically, etc.
# ClassifierConfig uses this to attach names to output probabilities.
CLASS_LABELS = [
    "acanthaceae", "ashwagandha", "asian ginseng", "astragalus", "basil", "birch", "black cohosh", 
    "black haw", "black pepper", "black walnut", "burdock", "calendula", "california poppy", "catnip",
    "chamomile", "chaste tree tree", "chickweed", "comfrey", "coriander", "cramp bark", "cumin", "dandelion",
    "echinacea", "elder berry", "elder berry flower", "elecampane", "eleuthero", "fennel", "feverfew",
    "garlic", "ginger", "ginger root", "ginko leaf", "green tea", "holy basil tulsi", "hops", "lady's mantle",
    "lavendar", "lemon balm", "licorice root", "linden", "meadowsweet", "motherwort", "mullein", "nettle", "nutmeg",
    "oak", "oat", "orange", "oregano", "passionflower", "peppermint", "plantain leaf", "raspberry", "red clover",
    "reishi", "rosemary", "sage", "saw palmetto", "sheperd's purse", "shiitake", "skullcap", "splilanthes", "st. john's wort",
    "thyme", "tulsi", "turmeric", "valerian", "vervain", "white pine", "wild yam"
]

print(f"Total classes: {len(CLASS_LABELS)}")  # should print 70

# ---- Build & load model ----
# =============================================================================
# THE NORMALIZED RESNET WRAPPER
# =============================================================================
#
# WHY THIS WRAPPER EXISTS — THE CORE PROBLEM:
# ─────────────────────────────────────────────
# Training pipeline (Python):
#   Raw image → transforms.Normalize() → ResNet18 → raw scores
#   Python handles everything ✅ easy
#
# iOS app (Swift):
#   Camera takes photo → ??? → ResNet18 → raw scores
#   iOS doesn't run Python! Who does normalization? 😰
#
# OPTION 1 — Normalize in Swift (bad):
#   iOS developer has to write:
#     let mean = [0.485, 0.456, 0.406]
#     let std  = [0.229, 0.224, 0.225]
#     // apply to every pixel...
#   Easy to get wrong ❌
#   iOS dev needs to know ImageNet magic numbers ❌
#   Any mistake = wrong predictions silently ❌
#
# OPTION 2 — Bake it INTO the model (your approach ✅):
#   iOS developer just does:
#     let result = model.prediction(image: cameraPhoto)
#   Normalization happens automatically inside the model ✅
#   iOS dev doesn't need to know anything about ImageNet ✅
#   Impossible to forget or get wrong ✅
#
# THE WRAPPER DOES THREE THINGS:
#   1. Normalization  → (x - mean) / std    baked in ✅
#   2. ResNet18       → the actual model     baked in ✅
#   3. Softmax        → F.softmax(x, dim=1)  baked in ✅
#
# COMPLETE DATA FLOW:
#   Raw iOS camera pixels (0-255)
#         ↓  scale=1/255 (in ct.convert)
#   Pixels scaled to (0-1)
#         ↓  (x - mean) / std (in forward)
#   Normalized pixels centered around zero
#         ↓  ResNet18 backbone (frozen layers 1-17)
#   512 meaningful herb features
#         ↓  Your trained fc layer (layer 18)
#   70 raw scores (logits — can be negative!)
#         ↓  F.softmax
#   {"chamomile": 0.85, "basil": 0.03...}  ← iOS shows this! ✅
class NormalizedResNet(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.model = resnet

        # register_buffer — permanently glues constants to the model
        # ─────────────────────────────────────────────────────────
        # WITHOUT register_buffer (storing as regular variable):
        #   self.mean = torch.tensor([0.485, 0.456, 0.406])
        #
        #   Problem 1 — model moves to GPU:
        #     model.to('mps')   → model moves to Mac GPU
        #     self.mean         → stays on CPU! 💀
        #     Can't subtract CPU tensor from GPU tensor → crash!
        #
        #   Problem 2 — model gets saved:
        #     torch.save(model.state_dict())
        #     self.mean NOT included in save! 😱
        #     Load model later → mean is gone!
        #
        #   Problem 3 — model exported to CoreML:
        #     self.mean NOT included in export
        #     iOS gets model without normalization values! 💀
        #
        # WITH register_buffer (your approach ✅):
        #   model.to('mps')  → mean automatically moves to GPU ✅
        #   torch.save()     → mean saved with model ✅
        #   ct.convert()     → mean baked into CoreML model ✅
        #   model.to('cpu')  → mean moves back to CPU ✅
        #   Mean is GLUED to the model — travels everywhere with it!
        #
        # DIFFERENCE FROM REGULAR PARAMETERS (nn.Linear weights):
        #   Regular parameters:  travel ✅  saved ✅  move devices ✅  UPDATED during training ✅
        #   register_buffer:     travel ✅  saved ✅  move devices ✅  NEVER updated (constants) ❌
        #
        # Mean and std are CONSTANTS — not learned, not updated, not weights.
        # Just fixed numbers that need to travel with the model everywhere.
        # Think of the model as a suitcase:
        #   nn.Linear weights = clothes (main contents, always packed)
        #   register_buffer   = passport glued inside suitcase cover
        #   Passport travels everywhere ✅  Never changes at customs ✅
        #   But without it → model can't work in production 💀
        #
        # WHY THESE SPECIFIC VALUES (0.485, 0.456, 0.406)?
        # ──────────────────────────────────────────────────
        # These are the mean RGB pixel values of ALL 1.2M ImageNet images:
        #   Average Red   pixel across 1.2M images = 0.485
        #   Average Green pixel across 1.2M images = 0.456
        #   Average Blue  pixel across 1.2M images = 0.406
        #
        # WHY APPLY MEAN AND STANDARD DEVIATION AT ALL?
        # ───────────────────────────────────────────────
        # The core problem: the same herb looks completely different under
        # different lighting conditions:
        #
        #   Photo 1: taken indoors, dim lighting
        #     Pixel values: [0.1, 0.15, 0.12...]  ← all very low
        #
        #   Photo 2: taken outdoors, bright sun
        #     Pixel values: [0.9, 0.85, 0.92...]  ← all very high
        #
        # Both photos are chamomile — but pixels look completely different!
        # Without normalization ResNet18 might think they're different herbs! 😱
        #
        # After (x - mean) / std:
        #   Photo 1 (dim):    [-1.5, -1.3, -1.6...]  ← centered
        #   Photo 2 (bright): [1.8,  1.7,  1.9...]   ← centered
        #   Chamomile FEATURES (shape, texture) still visible in BOTH ✅
        #   Lighting difference largely cancelled out ✅
        #   ResNet18 focuses on STRUCTURE not BRIGHTNESS ✅
        #
        # Your 3,500 herb photos came from:
        #   Different phones      → different colour profiles
        #   Different lighting    → different brightness
        #   Different backgrounds → different colour casts
        #   Different times of day → different warmth
        # Without normalization: model confuses lighting with herb identity 😱
        # With normalization:    model focuses on herb structure and shape ✅
        #
        # TWO STEPS OF NORMALIZATION:
        #   Step 1 — Scale [0,255] to [0,1]: scale=1/255 in ct.convert()
        #     [45, 120, 30] ÷ 255 = [0.176, 0.471, 0.118]
        #
        #   Step 2 — Center around zero:  (x - mean) / std in forward()
        #     Red:   (0.176 - 0.485) / 0.229 = -1.35
        #     Green: (0.471 - 0.456) / 0.224 =  0.07
        #     Blue:  (0.118 - 0.406) / 0.225 = -1.28
        #     Result: [-1.35, 0.07, -1.28] ← centered around zero ✅
        #
        # MEAN subtraction = moving everything to center
        # STD division     = making everything the same scale
        #
        # Simple analogy — comparing student heights across countries:
        #   Raw heights: Dutch=186cm, Japanese=168cm (different averages!)
        #   Without normalization: model thinks Dutch is "bigger" in every way
        #   After normalization:   Dutch=+0.8 (above Dutch avg), Japanese=+0.6
        #   NOW we can fairly compare their relative heights ✅
        #   Same idea with herb images across different lighting conditions.
        #
        # .view(1,3,1,1) — reshapes for broadcasting:
        #   Original: [3]          → just 3 numbers
        #   After:    [1, 3, 1, 1] → batch, channels, height, width
        #   Herb image: [16, 3, 224, 224]
        #   Broadcasting: mean [1,3,1,1] auto-expands to [16,3,224,224]
        #   Subtracts correct mean from EACH channel across ALL pixels ✅

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):

        # ---- Convert ----

        # Step 1: Normalize — center pixels around zero
        # iOS sends raw pixels (0-1 after scale=1/255):
        #   [0.176, 0.471, 0.118...]
        # After (x - mean) / std:
        #   [-1.35, 0.07, -1.28...]  ← ResNet18's language ✅
        x = (x - self.mean) / self.std

        # Step 2: ResNet18 forward pass
        # Normalized pixels flow through all 18 layers
        # Output: 70 RAW scores called LOGITS
        # These can be ANY number — including negative!
        # Example:
        #   chamomile: -2.3  ← negative! (this is what iOS was showing before)
        #   basil:      0.5
        #   lavender:   3.1  ← highest = most likely
        #   nettle:    -0.8
        x = self.model(x)

        # Step 3: Softmax — converts logits to probabilities
        # THIS is what fixed the negative numbers in your iOS app!
        #
        # Before softmax (raw logits):
        #   chamomile: -2.3,  basil: 0.5,  lavender: 3.1,  nettle: -0.8
        #
        # After softmax (probabilities):
        #   chamomile: 0.02 (2%),  basil: 0.08 (8%),  lavender: 0.85 (85%)
        #   Always sums to 1.0 (100%) ✅
        #
        # HOW SOFTMAX WORKS (the math):
        #   Uses e^x (exponential) for each value:
        #     e^(-2.3) = 0.10  ← small  (negative logit)
        #     e^(0.5)  = 1.65
        #     e^(3.1)  = 22.2  ← large  (positive logit)
        #     e^(-0.8) = 0.45
        #   Total = 0.10 + 1.65 + 22.2 + 0.45 = 24.4
        #   Divide each by total:
        #     chamomile: 0.10/24.4 = 0.4%
        #     lavender:  22.2/24.4 = 91%  ✅ highest logit = highest prob
        #
        # WHY dim=1?
        #   Output shape: [1, 70] → batch=1, classes=70
        #   dim=1 = apply softmax across the 70 CLASSES (not across batch)
        #   For each image → its 70 scores → convert to 70 probabilities ✅
        #
        # WHY NOT softmax during training?
        #   CrossEntropyLoss applies softmax INTERNALLY during training
        #   Adding softmax again = double softmax = wrong results! ❌
        #   iOS export has no CrossEntropyLoss → must add softmax explicitly ✅
        #
        # BEFORE wrapper: iOS showed [-2.3, 0.5, 3.1, -0.8...] 😱
        # AFTER wrapper:  iOS shows  "Lavender — 85% confidence" 🌿✅
        return F.softmax(x, dim=1)


# =============================================================================
# STEP 1: REBUILD THE TRAINED MODEL
# =============================================================================
#
# These 3 lines together do ONE thing:
# Rebuild your trained herb model from scratch so it can be wrapped and exported.
#
# WHY rebuild? Training script ended → model object disappeared from memory.
# Only my_herbs_model.pth survived on disk. Must rebuild structure THEN load weights.
#
# Think of it like rebuilding a custom racing car:
#   Line 1: Get standard car frame     (empty ResNet18 shell)
#   Line 2: Install 70-output gearbox  (correct fc layer shape)
#   Line 3: Install YOUR custom engine (trained weights from .pth)
#   Result: YOUR racing car, ready to race (export to iOS) 🏎️

# Line 1: Empty ResNet18 shell
# weights=None → DON'T download ImageNet weights (wasted bandwidth!)
# We're about to OVERWRITE all weights anyway from our .pth file
# Training used weights='IMAGENET1K_V1' but that was only needed once.
backbone = models.resnet18(weights=None)

# Line 2: Set correct output shape
# models.resnet18() creates 1000 output classes (ImageNet default)
# Must match EXACTLY what was used during training
# Otherwise load_state_dict will CRASH — wrong shape! ❌
# Training fc:    nn.Linear(512, 70) ← saved these weights
# Conversion fc:  nn.Linear(512, 70) ← must match to load them ✅
backbone.fc = nn.Linear(512, 71)

# Line 3: Load YOUR trained herb weights
# torch.load()          → reads my_herbs_model.pth from disk
# map_location='cpu'    → load onto CPU regardless of how it was saved
#                         (you trained on Mac MPS GPU → weights saved as MPS tensors)
#                         (converting on any machine → safely load to CPU first)
#                         (without this → might crash on different hardware ❌)
# load_state_dict()     → stuffs the 51,000 trained weights into the empty backbone
#                         {'fc.weight': tensor([[0.003, 0.012...]]),
#                          'fc.bias':   tensor([0.001, 0.002...])}
backbone.load_state_dict(torch.load('my_herbs_best.pth', map_location='cpu'))


# =============================================================================
# STEP 2: WRAP MODEL FOR IOS
# =============================================================================

# Combine backbone + normalization + softmax into one self-contained object
# backbone          = empty ResNet18 shell + YOUR 51,000 trained weights
# NormalizedResNet  = wrapper adding: mean/std baked in + softmax baked in
# Result: one object that takes raw iOS pixels and returns herb probabilities
model = NormalizedResNet(backbone)

# Switch from training mode to evaluation mode
# TWO modes in PyTorch:
#   model.train() ← training mode    (used in your training loop)
#   model.eval()  ← evaluation mode  (used for export and inference)
#
# What changes between modes:
#
#   Dropout layers:
#     Training mode:   randomly switches OFF some neurons (prevents overfitting)
#     Evaluation mode: ALL neurons active (want full power for predictions) ✅
#
#   BatchNorm layers:
#     Training mode:   calculates mean/std from CURRENT batch
#     Evaluation mode: uses STORED mean/std from training ✅ (consistent)
#
# Without model.eval():
#   Dropout randomly switches off neurons every scan
#   SAME herb photo → DIFFERENT results every time! 😱
#
# With model.eval():
#   ALL neurons active, fully consistent
#   SAME herb photo → SAME result every time ✅
#
# Real world analogy:
#   model.train() = doctor in LEARNING mode (takes risks, makes mistakes, learns)
#   model.eval()  = doctor in PRACTICE mode (consistent, reliable, no random decisions)
model.eval()


# =============================================================================
# STEP 3: TRACE THE MODEL (Record Operations as Static Graph)
# =============================================================================
#
# THE PROBLEM: iOS doesn't run Python! CoreML needs a STATIC description
# of the model, not live Python code.
#
# SOLUTION: torch.jit.trace watches the model process one example image
# and records every single mathematical operation:
#   "I saw you: subtract mean → divide by std → conv2d → relu → conv2d →
#    ... (hundreds of ops) ... → linear → softmax
#    I'll record ALL of this without needing Python to run it!"
#
# Real world analogy — teaching someone a recipe:
#   Option 1 (without tracing): give them the recipe BOOK (Python code)
#     They need to READ Python → if they don't speak it → can't cook! ❌
#   Option 2 (with tracing): cook the dish ONCE while they watch and record
#     Now ANYONE can follow the video — no recipe book, no Python needed ✅
#   torch.jit.trace = recording that cooking video
#
# WHAT traced_model contains:
#   ✅ All 11M ResNet18 weights (frozen backbone)
#   ✅ Your 51,000 trained fc weights
#   ✅ Mean and std normalization constants
#   ✅ Every mathematical operation in exact order
#   ✅ Input/output shapes
#   ❌ No Python code required to run it

# Fake image — just random pixels with correct shape
# Content doesn't matter at all — shape is ALL that matters
# [1, 3, 224, 224] = 1 image, RGB channels, 224x224 pixels
# Like a test car driving through a factory to map out the assembly line
example_input = torch.rand(1, 3, 224, 224)

# Feed example through model → record every operation → self-contained graph
traced_model = torch.jit.trace(model, example_input)


# =============================================================================
# STEP 4: CONVERT TO COREML AND SAVE
# =============================================================================
#
# ct.convert() translates the traced PyTorch graph into Apple's CoreML format.
#
# THREE PARAMETERS:
#
# 1. traced_model
#    The static graph recorded by torch.jit.trace
#    Contains all operations + all weights
#    coremltools reads this and translates to CoreML format ✅
#
# 2. inputs=[ct.ImageType(name="image", shape=..., scale=1/255.0)]
#    Tells CoreML what iOS feeds IN:
#      name="image"   → what Swift calls this input in code
#      shape=[1,3,224,224] → expected input dimensions
#      scale=1/255.0  → AUTOMATICALLY divide pixels by 255 before processing
#                        iOS camera gives integers 0-255
#                        Model expects floats 0-1
#                        CoreML handles this conversion invisibly ✅
#                        iOS app just passes raw camera photo — nothing extra needed
#
# 3. classifier_config=ct.ClassifierConfig(CLASS_LABELS)
#    Attaches herb names to output probabilities:
#      Without: iOS output = [0.02, 0.85, 0.03...] ← meaningless indices 😱
#      With:    iOS output = {"chamomile": 0.85, "basil": 0.02...} ✅
#    CoreML automatically maps index 0→'acanthaceae', index 14→'chamomile' etc.
#
# THE COMPLETE iOS USER EXPERIENCE THIS ENABLES:
#   User points camera at chamomile flower
#         ↓
#   iOS captures raw photo pixels [0-255]
#         ↓
#   CoreML: scale by 1/255         → [0-1]          (scale parameter)
#   CoreML: subtract mean, div std → centered        (register_buffer in forward)
#   CoreML: ResNet18 18 layers     → 512 features    (traced backbone)
#   CoreML: fc layer               → 70 raw scores   (traced fc layer)
#   CoreML: softmax                → probabilities   (traced forward method)
#   CoreML: map to names           → herb labels     (ClassifierConfig)
#         ↓
#   App displays: "Chamomile — 85% confidence" 🌼
#
# All triggered by ONE line in Swift:
#   let result = model.prediction(image: cameraPhoto)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="image", shape=example_input.shape, scale=1/255.0)],
    classifier_config=ct.ClassifierConfig(CLASS_LABELS)
)

# Save as .mlpackage — Apple's CoreML format
# Contains EVERYTHING iOS needs:
#   ✅ All model weights (11M backbone + 51K your trained weights)
#   ✅ Normalization constants (mean/std)
#   ✅ Input specification (224x224 RGB image)
#   ✅ scale=1/255 preprocessing
#   ✅ Output specification (70 herb probabilities with names)
#   ✅ Zero extra code needed in iOS app
# Drag into Xcode → works immediately 🎯
mlmodel.save('plantsnap_v1.mlpackage')
print("✅ iOS ready: my_herbs.mlpackage (Drag to Xcode!)")

