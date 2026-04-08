import requests
import sys

BASE_URL = "https://computer-vision-yin8.onrender.com"
# BASE_URL = "http://localhost:8000"  # uncomment for local

def flush_db():
    print("\n⚠️  PlantSnap Database Reset Tool")
    print("─" * 40)
    
    # First show current stats
    r = requests.get(f"{BASE_URL}/feedback/stats")
    if r.status_code == 200:
        stats = r.json()
        print(f"Current state:")
        print(f"  total_feedback:   {stats['total_feedback']}")
        print(f"  images_collected: {stats['images_collected']}")
        print(f"  storage_type:     {stats['storage_type']}")
    
    print("\n⚠️  This will DELETE all feedback records!")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() != "yes":
        print("❌ Cancelled — nothing deleted!")
        sys.exit(0)
    
    # Flush
    r = requests.post(f"{BASE_URL}/admin/flush-db")
    if r.status_code == 200:
        print("✅ Database flushed successfully!")
    else:
        print(f"❌ Failed: {r.status_code} — {r.text}")

if __name__ == "__main__":
    flush_db()