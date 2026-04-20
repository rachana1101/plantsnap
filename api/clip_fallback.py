# At startup (runs once):
#   1. Loads CLIP model (ViT-B/32)
#      → a model that understands BOTH images AND text

#   2. Reads your 71 herb folder names
#      → "acanthaceae", "chamomile", "lavender"...

#   3. Converts all 71 names into 512-dim vectors
#      → each herb is now a point in "CLIP space"
#      → stored in memory, ready to compare

# At inference (runs per image):
#   4. Takes an image → converts to 512-dim vector
#      → image is now a point in the SAME space

#   5. Compares image vector to ALL 71 herb vectors
#      → dot product = cosine similarity
#      → higher score = better match

#   6. Returns top 3 matches with scores
#      → acanthaceae: 0.2825 (best match)
#      → skullcap:    0.2637
#      → vervain:     0.2507

import torch
import clip
from PIL import Image
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def load_herb_names(data_dir: Path) -> list[str]:
    herbs = sorted([
        d.name.replace("_", " ").replace("-", " ")
        for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    print(f"✅ Loaded {len(herbs)} herb classes")
    return herbs

HERB_NAMES = load_herb_names(DATA_DIR)

# Encode all herb names ONCE at startup
# This is expensive (~2 sec) so we do it once, not per request
text_tokens = clip.tokenize(HERB_NAMES).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"✅ Encoded {len(HERB_NAMES)} herb text vectors")
print(f"   Shape: {text_features.shape}")


# preprocess(Image.open(...))
#   → loads image, resizes to 224x224, normalises

# model.encode_image(image)
#   → converts image to 512-dim vector

# image_features @ text_features.T
#   → dot product of image vs ALL 71 herbs
#   → returns 71 similarity scores

# argsort(descending=True)[:3]
#   → top 3 most similar herbs
def clip_identify(image_path: str) -> dict:
    """When ResNet isn't sure, ask CLIP."""
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # Compare image to ALL 71 herb text vectors
        similarities = (image_features @ text_features.T).squeeze()
        
        # Top 3 matches
        top3_idx = similarities.argsort(descending=True)[:3]
    
    results = []
    for idx in top3_idx:
        results.append({
            "herb": HERB_NAMES[idx],
            "similarity": round(similarities[idx].item(), 4)
        })
    
    print(f"🌿 CLIP result: {results[0]['herb']} ({results[0]['similarity']})")
    return {
        "best_match": results[0],
        "top_3": results,
        "method": "clip_zero_shot"
    }

# Test on a real image from your val set
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        # Skip hidden files, get first herb folder
        herb_dirs = sorted([
            d for d in DATA_DIR.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ])
        first_herb = herb_dirs[0]
        images = sorted([
            f for f in first_herb.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        ])
        test_image = str(images[0])
    
    print(f"\n🔍 Testing: {test_image}")
    result = clip_identify(test_image)
    
    print(f"\n📊 Top 3 CLIP matches:")
    for r in result["top_3"]:
        print(f"   {r['herb']:30s}  {r['similarity']:.4f}")