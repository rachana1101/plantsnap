import Foundation
import UIKit

// MARK: - Feedback Model
struct HerbFeedback: Codable {
    let image_id:       String
    let predicted_herb: String
    let correct_herb:   String
    let confidence:     Float
    let device_id:      String
    let app_version:    String
    let image_base64:   String?
    let is_new_herb:    Bool
}

// MARK: - Feedback Service
class FeedbackService {
    
    static let shared = FeedbackService()
    
    private let apiURL   = "https://computer-vision-yin8.onrender.com/feedback"
    private let queueKey = "plantsnap_feedback_queue"
    private let deviceID = UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
    
    // ── Send or Queue ────────────────────────────────
    func sendFeedback(
        imageID:       String,
        predictedHerb: String,
        correctHerb:   String,
        confidence:    Float,
        image:         UIImage? = nil,
        isNewHerb:     Bool = false
    ) {
        // ── Convert image to base64 ──────────────────
            var imageBase64: String? = nil
            if let image = image,
               let imageData = image.jpegData(compressionQuality: 0.5) {
                imageBase64 = imageData.base64EncodedString()
                log("📸 Image attached: \(imageData.count / 1024)KB")
            } else {
                log("📭 No image attached")
            }
        
        let feedback = HerbFeedback(
            image_id:       imageID,
            predicted_herb: predictedHerb,
            correct_herb:   correctHerb,
            confidence:     confidence,
            device_id:      deviceID,
            app_version:    "1.0",
            image_base64:   imageBase64,
            is_new_herb:    isNewHerb
        )
        
        log("📤 Attempting to send feedback:")
        log("   image_id:       \(imageID)")
        log("   predicted_herb: \(predictedHerb)")
        log("   correct_herb:   \(correctHerb)")
        log("   confidence:     \(String(format: "%.2f", confidence))")
        log("   device_id:      \(deviceID)")
        log("   is_new_herb:    \(isNewHerb)")
        log("   has_image:      \(imageBase64 != nil)")
        
        Task {
            let success = await postFeedback(feedback)
            if !success {
                saveFeedbackToQueue(feedback)
                log("📦 No internet — saved to local queue")
                logQueueStatus()
            } else {
                log("✅ Feedback sent to API successfully!")
                await syncQueue()
            }
        }
    }
    
    // ── POST to API ──────────────────────────────────
    private func postFeedback(_ feedback: HerbFeedback) async -> Bool {
        guard let url  = URL(string: apiURL),
              let body = try? JSONEncoder().encode(feedback) else {
            log("❌ Failed to create request")
            return false
        }
        
        var request        = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json",
                         forHTTPHeaderField: "Content-Type")
        request.httpBody        = body
        request.timeoutInterval = 10
        
        log("🌐 Calling API: POST \(apiURL)")
        
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            
            if let json = try? JSONSerialization.jsonObject(with: data) {
                log("📥 API response (\(code)): \(json)")
            }
            
            if code == 200 {
                log("✅ API accepted feedback")
                return true
            } else {
                log("⚠️ API returned status \(code)")
                return false
            }
        } catch {
            log("❌ Network error: \(error.localizedDescription)")
            return false
        }
    }
    
    // ── Save to Queue ────────────────────────────────
    private func saveFeedbackToQueue(_ feedback: HerbFeedback) {
        var queue = loadQueue()
        queue.append(feedback)
        
        if let data = try? JSONEncoder().encode(queue) {
            UserDefaults.standard.set(data, forKey: queueKey)
            log("💾 Saved to UserDefaults[\"\(queueKey)\"]")
        }
    }
    
    // ── Load Queue ───────────────────────────────────
    private func loadQueue() -> [HerbFeedback] {
        guard let data  = UserDefaults.standard.data(forKey: queueKey),
              let queue = try? JSONDecoder().decode(
                            [HerbFeedback].self, from: data)
        else { return [] }
        return queue
    }
    
    // ── Sync Queue ───────────────────────────────────
    func syncQueue() async {
        let queue = loadQueue()
        guard !queue.isEmpty else {
            log("📭 Queue is empty — nothing to sync")
            return
        }
        
        log("🔄 Syncing queue: \(queue.count) item(s) pending")
        logQueueStatus()
        
        var remaining: [HerbFeedback] = []
        
        for (index, feedback) in queue.enumerated() {
            log("   Sending item \(index + 1)/\(queue.count):")
            log("   → \(feedback.predicted_herb) → \(feedback.correct_herb)")
            
            let success = await postFeedback(feedback)
            if success {
                log("   ✅ Item \(index + 1) synced!")
            } else {
                log("   ❌ Item \(index + 1) failed — keeping in queue")
                remaining.append(feedback)
            }
        }
        
        // Save remaining failures back
        if let data = try? JSONEncoder().encode(remaining) {
            UserDefaults.standard.set(data, forKey: queueKey)
        }
        
        let synced = queue.count - remaining.count
        log("📊 Sync complete: \(synced) sent, \(remaining.count) remaining")
        logQueueStatus()
    }
    
    // ── Queue Status ─────────────────────────────────
    private func logQueueStatus() {
        let queue = loadQueue()
        log("📋 Queue status: \(queue.count) item(s) in UserDefaults[\"\(queueKey)\"]")
        
        if queue.isEmpty {
            log("   (empty)")
        } else {
            for (i, item) in queue.enumerated() {
                log("   [\(i+1)] \(item.predicted_herb) → \(item.correct_herb) " +
                    "(\(String(format: "%.0f", item.confidence * 100))% conf)")
            }
        }
    }
    
    // ── Logger ───────────────────────────────────────
    private func log(_ message: String) {
        let timestamp = DateFormatter.localizedString(
            from: Date(),
            dateStyle: .none,
            timeStyle: .medium
        )
        print("[PlantSnap \(timestamp)] \(message)")
    }
}
