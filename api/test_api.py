import requests
import json
import sys
import time

from pathlib import Path

API_DIR   = Path(__file__).parent
DATA_DIR  = API_DIR.parent / "data" / "raw"
RUN_CLIP_TESTS = True   # needs torch + clip installed locally

# ── CONFIG ────────────────────────────────────────────
#BASE_URL = "https://computer-vision-yin8.onrender.com"           
BASE_URL = "http://localhost:8000"  # uncomment for local testing

PASS = "✅"
FAIL = "❌"
results = []

def test(name, passed, details=""):
    status = PASS if passed else FAIL
    results.append(passed)
    print(f"{status} {name}")
    if not passed and details:
        print(f"   → {details}")

# ── TESTS ─────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/")
    test("Health check returns 200",
         r.status_code == 200)
    test("Health check has status field",
         "status" in r.json())

def test_version():
    r = requests.get(f"{BASE_URL}/version")
    test("Version endpoint returns 200",
         r.status_code == 200)
    data = r.json()
    test("Version has model_version field",
         "model_version" in data)
    test("Version has update_available field",
         "update_available" in data)

def test_feedback_valid():
    payload = {
        "image_id":       "test_chamomile_001",
        "predicted_herb": "basil",
        "correct_herb":   "chamomile",
        "confidence":     0.45,
        "device_id":      "test_device",
        "app_version":    "1.0"
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    test("POST /feedback returns 200",
         r.status_code == 200)
    data = r.json()
    test("Feedback response has id",
         "id" in data)
    test("Feedback response has correct_herb",
         data.get("correct_herb") == "chamomile")
    test("Feedback response has predicted_herb",
         data.get("predicted_herb") == "basil")

def test_feedback_high_confidence():
    """High confidence feedback — image should NOT be saved"""
    payload = {
        "image_id":       "test_lavender_001",
        "predicted_herb": "lavender",
        "correct_herb":   "lavender",
        "confidence":     0.95,  # HIGH — no image needed
        "device_id":      "test_device",
        "app_version":    "1.0"
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    test("High confidence feedback accepted",
         r.status_code == 200)

def test_feedback_missing_fields():
    """Should fail with 422 validation error"""
    payload = {
        "image_id": "test_002"
        # missing required fields!
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    test("Missing fields returns 422",
         r.status_code == 422,
         f"Got {r.status_code} instead of 422")

def test_metrics():
    payload = {
        "herb_name":   "chamomile",
        "confidence":  0.87,
        "was_correct": 1,
        "device_id":   "test_device"
    }
    r = requests.post(f"{BASE_URL}/metrics", json=payload)
    test("POST /metrics returns 200",
         r.status_code == 200)
    test("Metrics response has status",
         "status" in r.json())

def test_stats():
    r = requests.get(f"{BASE_URL}/feedback/stats")
    test("GET /feedback/stats returns 200",
         r.status_code == 200)
    data = r.json()
    test("Stats has total_feedback",
         "total_feedback" in data)
    test("Stats total is a number",
         isinstance(data.get("total_feedback"), int))

def test_invalid_confidence():
    """Confidence must be a float"""
    payload = {
        "image_id":       "test_003",
        "predicted_herb": "basil",
        "correct_herb":   "chamomile",
        "confidence":     "not_a_number",  # ← invalid!
        "device_id":      "test_device",
        "app_version":    "1.0"
    }
    r = requests.post(f"{BASE_URL}/feedback", json=payload)
    test("Invalid confidence type returns 422",
         r.status_code == 422,
         f"Got {r.status_code} instead of 422")
    

def wake_up_server():
    """Ping server and wait for it to wake up"""
    print("⏳ Waking up server...")
    for attempt in range(5):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=35)
            if r.status_code == 200:
                print("✅ Server is awake!")
                time.sleep(1)  # small buffer
                return
        except:
            pass
        print(f"   Attempt {attempt + 1}/5 — waiting...")
        time.sleep(5)
    print("⚠️ Server may still be waking up")    

def test_flush_db():
    """Test flush endpoint exists and works"""
    r = requests.post(f"{BASE_URL}/admin/flush-db")
    test("Flush DB endpoint returns 200",
         r.status_code == 200)
    test("Flush DB returns status",
         "status" in r.json())
    test("Flush DB returns deleted counts",
         "deleted_feedback" in r.json())    

def test_is_plant():
    """Local-only: calls clip_fallback.is_plant() directly (not over HTTP)"""
    try:
        from clip_fallback import is_plant
    except ImportError as e:
        test("CLIP plant check imports", False, str(e))
        return

    # 1. A non-plant photo should be rejected
    bus = API_DIR / "bus.jpg"
    r = is_plant(str(bus))
    test("is_plant returns expected keys",
         {"is_plant", "plant_score", "best_prompt", "threshold"} <= r.keys(),
         f"got {list(r.keys())}")
    test("Bus photo is NOT a plant",
         r["is_plant"] is False,
         f"score={r['plant_score']}, best_prompt='{r['best_prompt']}'")

    # 2. A real herb photo from the training data should pass
    basil = DATA_DIR / "basil" / "Image_1.jpeg"
    if basil.exists():
        r = is_plant(str(basil))
        test("Basil photo IS a plant",
             r["is_plant"] is True,
             f"score={r['plant_score']}, best_prompt='{r['best_prompt']}'")
    else:
        test("Basil test image found", False, f"missing {basil}")

    # 3. Score is a probability
    test("Plant score is between 0 and 1",
         0.0 <= r["plant_score"] <= 1.0,
         f"score={r['plant_score']}")    

# ── RUN ALL TESTS ──────────────────────────────────────
if __name__ == "__main__":
    print(f"\n🌿 PlantSnap API Test Suite")
    print(f"📡 Testing: {BASE_URL}")
    print("─" * 40)

    wake_up_server()

    test_health()
    test_version()
    test_feedback_valid()
    test_feedback_high_confidence()
    test_feedback_missing_fields()
    test_metrics()
    test_stats()
    test_invalid_confidence()
    test_is_plant()

    print("─" * 40)
    passed = sum(results)
    total  = len(results)
    print(f"\n{'🎉' if passed == total else '⚠️ '} {passed}/{total} tests passed")

    # Exit with error code if any test failed
    # (useful for CI/CD pipelines!)
    sys.exit(0 if passed == total else 1)

