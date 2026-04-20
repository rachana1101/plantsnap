import os
from pathlib import Path

# ── Config ────────────────────────────────────────────
DATA_DIR = "../data/raw"  # your herb folders

def clean_folder_names():
    data = Path(DATA_DIR)
    
    renamed  = 0
    skipped  = 0
    
    print("🌿 Cleaning herb folder names...")
    print("─" * 40)
    
    for folder in sorted(data.iterdir()):
        if not folder.is_dir():
            continue
            
        original = folder.name
        
        # Clean: lowercase + remove "herb plant", "herb", "plant"
        cleaned = original.lower()
        cleaned = cleaned.replace(" herb plant", "")
        cleaned = cleaned.replace(" herb", "")
        cleaned = cleaned.replace(" plant", "")
        cleaned = cleaned.strip()
        
        if cleaned == original:
            print(f"⏭️  {original} (no change)")
            skipped += 1
            continue
        
        new_path = data / cleaned
        
        # Check for conflicts
        if new_path.exists():
            print(f"⚠️  CONFLICT: {original} → {cleaned} already exists!")
            continue
        
        folder.rename(new_path)
        print(f"✅ {original} → {cleaned}")
        renamed += 1
    
    print("─" * 40)
    print(f"✅ Renamed: {renamed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"\nVerify your folders:")
    print(f"  ls {DATA_DIR}")

if __name__ == "__main__":
    clean_folder_names()