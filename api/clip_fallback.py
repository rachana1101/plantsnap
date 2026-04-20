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
from pathlib import Path

# Don't load anything yet — wait until first call
_model = None
_preprocess = None
_text_features = None
_herb_names = None

def _load_clip():
    """Load CLIP model lazily — only on first /clip request"""
    global _model, _preprocess, _text_features, _herb_names
    
    if _model is not None:
        return  # already loaded
    
    print("🔄 Loading CLIP model (first request)...")
    device = "cpu"  # Render = CPU only
    _model, _preprocess = clip.load("ViT-B/32", device=device)
    _model.eval()
    
    # Load herb names from data folder
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    if data_dir.exists():
        _herb_names = sorted([
            d.name.replace("_", " ").replace("-", " ")
            for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
    else:
        _herb_names = ["chamomile", "lavender", "peppermint"]
    
    # Encode herb names
    text_tokens = clip.tokenize(_herb_names).to(device)
    with torch.no_grad():
        _text_features = _model.encode_text(text_tokens)
        _text_features /= _text_features.norm(dim=-1, keepdim=True)
    
    print(f"✅ CLIP loaded: {len(_herb_names)} herbs")

def clip_identify(image_path: str) -> dict:
    _load_clip()  # loads only on first call
    
    device = "cpu"
    image = _preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = _model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ _text_features.T).squeeze()
        top3_idx = similarities.argsort(descending=True)[:3]
    
    results = []
    for idx in top3_idx:
        results.append({
            "herb": _herb_names[idx],
            "similarity": round(similarities[idx].item(), 4)
        })
    
    return {
        "best_match": results[0],
        "top_3": results,
        "method": "clip_zero_shot"
    }