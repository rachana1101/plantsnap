"""
download_herbs.py

Downloads images from iNaturalist for each herb folder in data/raw/
Automatically reads YOUR folder names — no hardcoding needed!

Usage:
    cd ~/techProjects/plantsnap/ml
    source venv/bin/activate
    pip install requests
    python download_herbs.py
"""

import requests
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────
DATA_DIR = "/Users/rachana_gupta/techProjects/plantsnap/data/raw"
TARGET   = 150   # target images per herb

# ── Name mappings ─────────────────────────────────────
# iNaturalist uses scientific or well-known common names
# Maps YOUR folder name → iNaturalist taxon name
# Add more here if a herb shows 0 downloads!

NAME_MAP = {
    # Common name issues
    "asian ginseng":        "Panax ginseng",
    "american ginseng":     "Panax quinquefolius",
    "st. john's wort":      "Hypericum perforatum",
    "st johns wort":        "Hypericum perforatum",
    "sheperd's purse":      "Capsella bursa-pastoris",
    "shepherds purse":      "Capsella bursa-pastoris",
    "saw palmetto":         "Serenoa repens",
    "wild yam":             "Dioscorea villosa",
    "white pine":           "Pinus strobus",
    "red clover":           "Trifolium pratense",
    "tulsi":                "Ocimum tenuiflorum",
    "ashwagandha":          "Withania somnifera",
    "astragalus":           "Astragalus membranaceus",
    "spilanthes":           "Acmella oleracea",
    "splilanthes":          "Acmella oleracea",
    "skullcap":             "Scutellaria lateriflora",
    "reishi":               "Ganoderma lucidum",
    "shiitake":             "Lentinula edodes",
    "valerian":             "Valeriana officinalis",
    "vervain":              "Verbena officinalis",
    "marshmallow":          "Althaea officinalis",
    "marshmallow root":     "Althaea officinalis",
    "holy basil":           "Ocimum tenuiflorum",
    "lemon balm":           "Melissa officinalis",
    "milk thistle":         "Silybum marianum",
    "cats claw":            "Uncaria tomentosa",
    "cat's claw":           "Uncaria tomentosa",
    "dong quai":            "Angelica sinensis",
    "gotu kola":            "Centella asiatica",
    "he shou wu":           "Reynoutria multiflora",
    "fo-ti":                "Reynoutria multiflora",
    "licorice":             "Glycyrrhiza glabra",
    "licorice root":        "Glycyrrhiza glabra",
    "burdock":              "Arctium lappa",
    "burdoc":               "Arctium lappa",
    "dandelion":            "Taraxacum officinale",
    "elderflower":          "Sambucus nigra",
    "elderberry":           "Sambucus nigra",
    "echinacea":            "Echinacea purpurea",
    "feverfew":             "Tanacetum parthenium",
    "black cohosh":         "Actaea racemosa",
    "blue cohosh":          "Caulophyllum thalictroides",
    "goldenseal":           "Hydrastis canadensis",
    "hawthorn":             "Crataegus monogyna",
    "kava":                 "Piper methysticum",
    "passionflower":        "Passiflora incarnata",
    "passion flower":       "Passiflora incarnata",
    "rhodiola":             "Rhodiola rosea",
    "schisandra":           "Schisandra chinensis",
    "shatavari":            "Asparagus racemosus",
    "nettle":               "Urtica dioica",
    "stinging nettle":      "Urtica dioica",
    "plantain leaf":        "Plantago major",
    "plantain":             "Plantago major",
    "catnip herb":          "Nepeta cataria",
    "catnip":               "Nepeta cataria",
    "peppermint":           "Mentha x piperita",
    "spearmint":            "Mentha spicata",
    "chamomile":            "Matricaria chamomilla",
    "german chamomile":     "Matricaria chamomilla",
    "roman chamomile":      "Chamaemelum nobile",
    "calendula":            "Calendula officinalis",
    "lavender":             "Lavandula angustifolia",
    "rosemary":             "Salvia rosmarinus",
    "thyme":                "Thymus vulgaris",
    "sage":                 "Salvia officinalis",
    "oregano":              "Origanum vulgare",
    "basil":                "Ocimum basilicum",
    "turmeric":             "Curcuma longa",
    "ginger":               "Zingiber officinale",
    "acanthaceae":          "Acanthus mollis",
}

# ── Download function ─────────────────────────────────
def download_herb_images(herb_name: str):
    herb_dir  = Path(DATA_DIR) / herb_name
    herb_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images
    existing = len(
        list(herb_dir.glob("*.jpg")) +
        list(herb_dir.glob("*.jpeg")) +
        list(herb_dir.glob("*.png"))
    )
    needed = max(0, TARGET - existing)

    if needed == 0:
        print(f"⏭️  {herb_name}: already has {existing} images ✅")
        return existing

    # Use mapped name if available
    taxon_name = NAME_MAP.get(herb_name.lower(), herb_name)
    if taxon_name != herb_name:
        print(f"📥 {herb_name} → '{taxon_name}': has {existing}, need {needed} more...")
    else:
        print(f"📥 {herb_name}: has {existing}, need {needed} more...")

    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "taxon_name":    taxon_name,
                "photos":        True,
                "quality_grade": "research",
                "per_page":      min(needed + 20, 200),  # grab extra for dupes
                "order_by":      "votes",
                "photo_license": "cc-by,cc-by-nc,cc0"
            },
            timeout=30
        )
        r.raise_for_status()
    except Exception as e:
        print(f"  ❌ API error: {e}")
        return existing

    results    = r.json().get("results", [])
    downloaded = 0

    for obs in results:
        if downloaded >= needed:
            break

        if not obs.get("photos"):
            continue

        photo_url = obs["photos"][0]["url"].replace("square", "large")
        filename  = herb_dir / f"inat_{obs['id']}.jpg"

        if filename.exists():
            continue

        try:
            img = requests.get(photo_url, timeout=15)
            if img.status_code == 200:
                filename.write_bytes(img.content)
                downloaded += 1
        except Exception:
            pass

        time.sleep(0.2)  # be polite to iNaturalist ✅

    total = existing + downloaded
    if downloaded > 0:
        print(f"  ✅ Downloaded {downloaded} → total: {total}")
    else:
        print(f"  ⚠️  Got 0 — try adding '{herb_name}' to NAME_MAP")

    return total


# ── Main ─────────────────────────────────────────────
def main():
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        print(f"❌ Data folder not found: {DATA_DIR}")
        print(f"   Make sure you run from plantsnap/ml/ folder!")
        return

    # Auto-read YOUR folder names ✅
    herbs = [d.name for d in sorted(data_path.iterdir())
             if d.is_dir()]

    if not herbs:
        print(f"❌ No herb folders found in {DATA_DIR}")
        return

    print(f"🌿 Found {len(herbs)} herb folders")
    print(f"   Target: {TARGET} images each")
    print(f"   Source: iNaturalist (research grade, open license)")
    print("─" * 50)

    results = {}
    for herb in herbs:
        total = download_herb_images(herb)
        results[herb] = total
        time.sleep(0.3)

    # ── Summary ───────────────────────────────────────
    print("\n" + "─" * 50)
    print("📊 Final summary:")

    ok       = {h: c for h, c in results.items() if c >= TARGET}
    low      = {h: c for h, c in results.items() if 50 <= c < TARGET}
    very_low = {h: c for h, c in results.items() if c < 50}

    print(f"\n  ✅ At target ({TARGET}+): {len(ok)} herbs")
    
    if low:
        print(f"\n  ⚠️  Below target (50-{TARGET-1}): {len(low)} herbs")
        for h, c in sorted(low.items(), key=lambda x: x[1]):
            print(f"     {h}: {c} images")

    if very_low:
        print(f"\n  ❌ Very low (<50): {len(very_low)} herbs — consider removing or renaming!")
        for h, c in sorted(very_low.items(), key=lambda x: x[1]):
            mapped = NAME_MAP.get(h.lower(), "no mapping")
            print(f"     {h}: {c} images  (iNat name: {mapped})")

    total_images = sum(results.values())
    print(f"\n  Total images: {total_images}")
    print(f"\n✅ Done! Next steps:")
    print(f"   python split_dataset.py")
    print(f"   python train_herbs_mlflow.py")


if __name__ == "__main__":
    main()