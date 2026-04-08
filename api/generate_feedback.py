# generate_feedback.py
import requests
import random
import uuid

BASE_URL = "https://computer-vision-yin8.onrender.com"

# Your 70 herbs
herbs = [
    'acanthaceae', 'ashwagandha', 'asian ginseng', 'astragalus',
    'basil', 'birch', 'black cohosh', 'black haw', 'black pepper',
    'black walnut', 'burdock', 'calendula', 'california poppy',
    'catnip herb', 'chamomile', 'chaste tree', 'chickweed', 'comfrey',
    'coriander', 'cramp bark', 'cumin', 'dandelion', 'echinacea',
    'elder berry', 'elder berry flower', 'elecampane', 'eleuthero',
    'fennel', 'feverfew', 'garlic', 'ginger', 'ginger root',
    'ginko leaf', 'green tea', 'holy basil', 'hops', 'lady\'s mantle',
    'lavender', 'lemon balm', 'licorice root', 'linden', 'meadowsweet',
    'motherwort', 'mullein', 'nettle', 'nutmeg', 'oak', 'orange',
    'oregano', 'passionflower', 'peppermint', 'plantain leaf',
    'raspberry', 'red clover', 'reishi', 'rosemary', 'sage',
    'saw palmetto', 'shepherd\'s purse', 'shiitake', 'skullcap',
    'spilanthes', 'st. john\'s wort', 'thyme', 'tulsi', 'turmeric',
    'valerian', 'vervain', 'white pine', 'wild yam']

def generate_feedback(n=100):
    for i in range(n):
        predicted = random.choice(herbs)
        correct   = random.choice(herbs)
        
        payload = {
            "image_id":       str(uuid.uuid4()),
            "predicted_herb": predicted,
            "correct_herb":   correct,
            "confidence":     round(random.uniform(0.2, 0.95), 2),
            "device_id":      f"test_device_{random.randint(1,10)}",
            "app_version":    "1.0",
            "is_new_herb":    False
        }
        
        r = requests.post(f"{BASE_URL}/feedback", json=payload)
        print(f"[{i+1}/n] {predicted} → {correct}: {r.status_code}")

if __name__ == "__main__":
    generate_feedback(100)