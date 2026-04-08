import SwiftUI
import UIKit
import CoreML
import Vision

// MARK: - Model
struct HerbPrediction: Identifiable {
    let id = UUID()
    let rank: Int
    let name: String
    let confidence: Float
}

// MARK: - ContentView
struct ContentView: View {
    
    // Add at top of file, outside any struct:
    let HERB_NAMES: [String] = [
        "basil", "chamomile", "lavender", "rosemary",
        "thyme", "mint", "oregano", "sage", "parsley",
        "cilantro", "dill", "fennel", "lemon balm",
        "echinacea", "elderflower", "calendula",
        "St Johns wort", "valerian", "ashwagandha",
        // add all 70 of your herbs here!
    ]
    
    @State private var predictions: [HerbPrediction] = []
    @State private var showingCamera = false
    @State private var showingAlbum = false
    @State private var selectedImage: UIImage?
    @State private var isLoading = false
    @State private var errorMessage: String?
    @State private var currentImageID: String = UUID().uuidString
    @State private var showingFeedback: Bool  = false
    @State private var feedbackSubmitted: Bool = false

    var body: some View {
        ZStack {
            Color(red: 0.1, green: 0.25, blue: 0.1)
                .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 24) {

                    // MARK: Header
                    VStack(spacing: 6) {
                        Text("🌻")
                            .font(.system(size: 52))
                        Text("Trail Botanist")
                            .font(.system(size: 30, weight: .bold, design: .rounded))
                            .foregroundColor(.white)
                        Text("Identify plants on your hike")
                            .font(.subheadline)
                            .foregroundColor(.white.opacity(0.75))
                    }
                    .padding(.top, 40)

                    // MARK: Image Card
                    ZStack {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.1))
                            .shadow(color: .black.opacity(0.3), radius: 10, y: 4)

                        if let image = selectedImage {
                            Image(uiImage: image)
                                .resizable()
                                .scaledToFill()
                                .frame(height: 260)
                                .clipShape(RoundedRectangle(cornerRadius: 20))
                        } else {
                            VStack(spacing: 12) {
                                Image(systemName: "leaf.circle.fill")
                                    .font(.system(size: 64))
                                    .foregroundColor(.white.opacity(0.4))
                                Text("Spot something on the trail?")
                                    .foregroundColor(.white.opacity(0.6))
                                    .font(.callout)
                            }
                            .frame(height: 260)
                        }
                    }
                    .frame(height: 260)
                    .padding(.horizontal)

                    // MARK: Results Card
                    VStack(alignment: .leading, spacing: 16) {
                        HStack(spacing: 6) {
                            Image(systemName: "sparkles")
                                .foregroundColor(.white)
                            Text("Results")
                                .font(.headline)
                                .foregroundColor(.white)
                        }

                        if isLoading {
                            HStack {
                                Spacer()
                                VStack(spacing: 10) {
                                    ProgressView()
                                        .progressViewStyle(
                                            CircularProgressViewStyle(tint: .white)
                                        )
                                        .scaleEffect(1.3)
                                    Text("Identifying plant...")
                                        .font(.callout)
                                        .foregroundColor(.white.opacity(0.7))
                                }
                                Spacer()
                            }
                            .padding(.vertical, 8)

                        } else if let error = errorMessage {
                            Text(error)
                                .foregroundColor(Color(red: 1, green: 0.5, blue: 0.5))
                                .font(.callout)

                        } else if predictions.isEmpty {
                            Text("Take a photo to identify a plant")
                                .foregroundColor(.white.opacity(0.6))
                                .font(.callout)

                        } else {
                            ForEach(predictions) { herb in
                                HerbResultRow(herb: herb)
                            }
                            // Add inside your Results Card, after ForEach(predictions):
                            if !predictions.isEmpty {
                                FeedbackBanner(
                                    topPrediction: predictions[0],
                                    submitted:     $feedbackSubmitted,
                                    imageID:       currentImageID,
                                    selectedImage: selectedImage    // ← ADD
                                )
                                .padding(.top, 8)
                            }
                        }
                    }
                    .padding()
                    .background(Color.white.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    .shadow(color: .black.opacity(0.2), radius: 10, y: 4)
                    .padding(.horizontal)

                    // MARK: Buttons
                    HStack(spacing: 12) {
                        // Camera
                        Button {
                            showingCamera = true
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "camera.fill")
                                Text("Camera")
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                UIImagePickerController.isSourceTypeAvailable(.camera)
                                ? Color(red: 0.15, green: 0.38, blue: 0.15)
                                    : Color.gray.opacity(0.4)
                            )
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
                        }
                        .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))

                        // Album
                        Button {
                            showingAlbum = true
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "photo.on.rectangle")
                                Text("Album")
                                    .font(.headline)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(Color(red: 0.15, green: 0.38, blue: 0.15))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                            .shadow(color: .black.opacity(0.3), radius: 8, y: 4)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 40)
                }
            }
        }
        // ✅ Two separate sheets — hardcoded source type, never ambiguous
        .sheet(isPresented: $showingCamera) {
            ImagePicker(
                image: $selectedImage,
                predictions: $predictions,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                sourceType: .camera,
                currentImageID:    $currentImageID,
                feedbackSubmitted: $feedbackSubmitted
            )
        }
        .sheet(isPresented: $showingAlbum) {
            ImagePicker(
                image: $selectedImage,
                predictions: $predictions,
                isLoading: $isLoading,
                errorMessage: $errorMessage,
                sourceType: .photoLibrary,
                currentImageID:    $currentImageID,
                feedbackSubmitted: $feedbackSubmitted
            )
        }
    }
}

// MARK: - Herb Result Row
struct HerbResultRow: View {
    let herb: HerbPrediction

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("\(herb.rank).")
                    .font(.system(.callout, design: .rounded))
                    .foregroundColor(.white.opacity(0.5))
                    .frame(width: 20)
                Text(herb.name.capitalized)
                    .font(.system(.body, design: .rounded))
                    .fontWeight(herb.rank == 1 ? .bold : .regular)
                    .foregroundColor(herb.rank == 1 ? .white : .white.opacity(0.85))
                Spacer()
                Text("\(Int(herb.confidence * 100))%")
                    .font(.system(.callout, design: .rounded))
                    .fontWeight(.semibold)
                    .foregroundColor(herb.rank == 1 ? .white : .white.opacity(0.6))
            }

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.white.opacity(0.15))
                        .frame(height: 8)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(herb.rank == 1 ? Color.white : Color.white.opacity(0.4))
                        .frame(width: geo.size.width * CGFloat(herb.confidence), height: 8)
                }
            }
            .frame(height: 8)
        }
    }
}


// MARK: - Feedback Banner
// MARK: - Feedback Banner (3-Tier)
struct FeedbackBanner: View {
    let topPrediction:  HerbPrediction
    @Binding var submitted: Bool
    let imageID:        String
    let selectedImage:  UIImage?

    // State
    @State private var tier:          Int    = 0
    // 0 = initial question
    // 1 = tier 1 known herbs dropdown
    // 2 = tier 2 extended searchable list
    // 3 = tier 3 free text input

    @State private var selectedHerb:  String = ""
    @State private var searchText:    String = ""
    @State private var freeText:      String = ""
    @State private var freeTextError: String = ""

    // Filtered herbs for search
    var filteredHerbs: [String] {
        if searchText.isEmpty {
            return ALL_EXTENDED_HERBS
        }
        return ALL_EXTENDED_HERBS.filter {
            $0.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        VStack(spacing: 12) {

            // ── Submitted ────────────────────────────
            if submitted {
                HStack(spacing: 8) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Thanks! Helping PlantSnap grow 🌿")
                        .font(.callout)
                        .foregroundColor(.white)
                }
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.white.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 16))

            // ── Tier 0: Initial question ─────────────
            } else if tier == 0 {
                VStack(spacing: 10) {
                    Text("Is this correct?")
                        .font(.callout)
                        .foregroundColor(.white.opacity(0.8))

                    HStack(spacing: 12) {
                        // 👍 Yes
                        Button {
                            submitFeedback(correctHerb: topPrediction.name,
                                         isNewHerb: false)
                        } label: {
                            HStack {
                                Image(systemName: "hand.thumbsup.fill")
                                Text("Yes!")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.green.opacity(0.7))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }

                        // 👎 No → go to tier 1
                        Button {
                            withAnimation { tier = 1 }
                        } label: {
                            HStack {
                                Image(systemName: "hand.thumbsdown.fill")
                                Text("No")
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.red.opacity(0.6))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                        }
                    }
                }
                .padding()
                .background(Color.white.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 16))

            // ── Tier 1: Known 70 herbs dropdown ──────
            } else if tier == 1 {
                VStack(spacing: 10) {
                    HStack {
                        Text("🌿 Select from known herbs:")
                            .font(.callout)
                            .foregroundColor(.white.opacity(0.8))
                        Spacer()
                        // back button
                        Button("← Back") {
                            withAnimation { tier = 0 }
                        }
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                    }

                    Picker("Select herb", selection: $selectedHerb) {
                        Text("Choose herb...").tag("")
                        ForEach(KNOWN_HERBS.sorted(), id: \.self) { herb in
                            Text(herb.capitalized).tag(herb)
                        }
                    }
                    .pickerStyle(.menu)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.white.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .foregroundColor(.white)
                    .accentColor(.white)

                    // Submit tier 1
                    Button {
                        guard !selectedHerb.isEmpty else { return }
                        submitFeedback(correctHerb: selectedHerb,
                                      isNewHerb: false)
                    } label: {
                        Text("Submit")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(selectedHerb.isEmpty
                                ? Color.gray.opacity(0.4)
                                : Color(red: 0.15, green: 0.5, blue: 0.15))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .disabled(selectedHerb.isEmpty)

                    // Go to tier 2
                    Button {
                        withAnimation { tier = 2 }
                    } label: {
                        Text("Not in this list → Search 200+ herbs")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                            .underline()
                    }
                }
                .padding()
                .background(Color.white.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 16))

            // ── Tier 2: Extended searchable list ─────
            } else if tier == 2 {
                VStack(spacing: 10) {
                    HStack {
                        Text("🔍 Search all herbs:")
                            .font(.callout)
                            .foregroundColor(.white.opacity(0.8))
                        Spacer()
                        Button("← Back") {
                            withAnimation { tier = 1 }
                        }
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                    }

                    // Search bar
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.white.opacity(0.5))
                        TextField("Type herb name...", text: $searchText)
                            .foregroundColor(.white)
                            .autocapitalization(.none)
                    }
                    .padding(10)
                    .background(Color.white.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                    // Results list
                    if filteredHerbs.isEmpty {
                        Text("Not found → try typing it yourself")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))

                        Button {
                            freeText = searchText
                            withAnimation { tier = 3 }
                        } label: {
                            Text("Type it myself →")
                                .font(.caption)
                                .foregroundColor(.white.opacity(0.8))
                                .underline()
                        }

                    } else {
                        ScrollView {
                            VStack(spacing: 4) {
                                ForEach(filteredHerbs.prefix(8),
                                        id: \.self) { herb in
                                    Button {
                                        submitFeedback(
                                            correctHerb: herb,
                                            isNewHerb: !KNOWN_HERBS.contains(herb)
                                        )
                                    } label: {
                                        HStack {
                                            Text(herb.capitalized)
                                                .foregroundColor(.white)
                                            Spacer()
                                            if !KNOWN_HERBS.contains(herb) {
                                                Text("new")
                                                    .font(.caption2)
                                                    .padding(.horizontal, 6)
                                                    .padding(.vertical, 2)
                                                    .background(
                                                        Color.orange.opacity(0.6))
                                                    .clipShape(Capsule())
                                                    .foregroundColor(.white)
                                            }
                                        }
                                        .padding(.horizontal, 12)
                                        .padding(.vertical, 8)
                                        .background(Color.white.opacity(0.1))
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    }
                                }
                            }
                        }
                        .frame(maxHeight: 200)
                    }

                    // Go to tier 3
                    Button {
                        withAnimation { tier = 3 }
                    } label: {
                        Text("Still not found → type it myself")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                            .underline()
                    }
                }
                .padding()
                .background(Color.white.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 16))

            // ── Tier 3: Free text with validation ────
            } else if tier == 3 {
                VStack(spacing: 10) {
                    HStack {
                        Text("✏️ Type the herb name:")
                            .font(.callout)
                            .foregroundColor(.white.opacity(0.8))
                        Spacer()
                        Button("← Back") {
                            withAnimation { tier = 2 }
                        }
                        .font(.caption)
                        .foregroundColor(.white.opacity(0.6))
                    }

                    TextField("e.g. elderflower", text: $freeText)
                        .foregroundColor(.white)
                        .autocapitalization(.none)
                        .onChange(of: freeText) { newValue in
                            // Validate as user types
                            freeTextError = validateHerbName(newValue)
                            // Max 50 chars
                            if newValue.count > 50 {
                                freeText = String(newValue.prefix(50))
                            }
                        }
                        .padding(10)
                        .background(Color.white.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 10))

                    // Error message
                    if !freeTextError.isEmpty {
                        Text(freeTextError)
                            .font(.caption)
                            .foregroundColor(
                                Color(red: 1, green: 0.5, blue: 0.5))
                    }

                    // Character count
                    HStack {
                        Spacer()
                        Text("\(freeText.count)/50")
                            .font(.caption2)
                            .foregroundColor(.white.opacity(0.4))
                    }

                    Text("⚠️ Will be reviewed by our herb expert before use")
                        .font(.caption2)
                        .foregroundColor(.white.opacity(0.5))
                        .multilineTextAlignment(.center)

                    Button {
                        let error = validateHerbName(freeText)
                        if error.isEmpty && !freeText.isEmpty {
                            submitFeedback(
                                correctHerb: freeText.lowercased().trimmingCharacters(
                                    in: .whitespaces),
                                isNewHerb: true
                            )
                        }
                    } label: {
                        Text("Submit for Review")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(
                                freeText.isEmpty || !freeTextError.isEmpty
                                ? Color.gray.opacity(0.4)
                                : Color.orange.opacity(0.7))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                    .disabled(freeText.isEmpty || !freeTextError.isEmpty)
                }
                .padding()
                .background(Color.white.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 16))
            }
        }
    }

    // ── Validation ───────────────────────────────────
    private func validateHerbName(_ text: String) -> String {
        if text.isEmpty { return "" }

        // Letters, spaces, hyphens, apostrophes only
        let regex = "^[a-zA-Z\\s\\-']{2,50}$"
        if text.range(of: regex, options: .regularExpression) == nil {
            return "Letters and spaces only please!"
        }
        if text.count < 2 {
            return "Too short — minimum 2 characters"
        }
        return "" // valid!
    }

    // ── Submit ───────────────────────────────────────
    private func submitFeedback(correctHerb: String, isNewHerb: Bool) {
        FeedbackService.shared.sendFeedback(
            imageID:       imageID,
            predictedHerb: topPrediction.name,
            correctHerb:   correctHerb,
            confidence:    topPrediction.confidence,
            image:         selectedImage,
            isNewHerb:     isNewHerb
        )
        withAnimation { submitted = true }
    }
}
// MARK: - ImagePicker
struct ImagePicker: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Binding var predictions: [HerbPrediction]
    @Binding var isLoading: Bool
    @Binding var errorMessage: String?
    var sourceType: UIImagePickerController.SourceType
    @Binding var currentImageID: String
    @Binding var feedbackSubmitted: Bool

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = sourceType
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: ImagePicker

        init(_ parent: ImagePicker) { self.parent = parent }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]
        ) {
            guard let uiImage = info[.originalImage] as? UIImage else {
                picker.dismiss(animated: true)
                return
            }

            parent.image = uiImage
            parent.predictions = []
            parent.currentImageID    = UUID().uuidString
            parent.feedbackSubmitted = false
            parent.errorMessage = nil
            DispatchQueue.main.async { self.parent.isLoading = true }

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    guard let modelURL = Bundle.main.url(forResource: "my_herbs",
                                                         withExtension: "mlmodelc") else {
                        throw NSError(domain: "TrailBotanist", code: 1, userInfo: [
                            NSLocalizedDescriptionKey: "❌ Model not found in bundle"
                        ])
                    }

                    let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))

                    let request = VNCoreMLRequest(model: visionModel) { request, _ in
                        guard let results = request.results as? [VNClassificationObservation],
                              !results.isEmpty else { return }

                        let top3 = results.prefix(3).enumerated().map { i, r in
                            HerbPrediction(rank: i + 1, name: r.identifier, confidence: r.confidence)
                        }
                        DispatchQueue.main.async {
                            self.parent.predictions = top3
                            self.parent.isLoading = false
                        }
                    }
                    request.imageCropAndScaleOption = .centerCrop

                    guard let cgImage = uiImage.cgImage else { return }
                    try VNImageRequestHandler(cgImage: cgImage, options: [:]).perform([request])

                } catch {
                    DispatchQueue.main.async {
                        self.parent.errorMessage = "❌ \(error.localizedDescription)"
                        self.parent.isLoading = false
                    }
                }
            }
            picker.dismiss(animated: true)
        }
    }
}

#Preview { ContentView() }

