# Rice Disease Classifier — Full Explanation

---

## The Big Picture

We built a system that looks at a photo of a rice leaf and tells you which disease it has — or whether it is healthy. It can recognise 6 conditions:

- Bacterial Blight
- Blast
- Brown Spot
- Tungro
- Healthy Rice Leaf
- Hispa

The system is built on a deep learning model called a **Convolutional Neural Network (CNN)**, specifically using a well-known architecture called **ResNet-50**.

---

## What is ResNet-50?

Imagine you want to teach someone to recognise diseases in rice. Instead of starting from scratch, you hire an expert who already knows how to look at images in general — they can already identify edges, textures, shapes, colours. You then train them specifically on rice diseases on top of that existing knowledge.

That is exactly what we did. ResNet-50 is a neural network that was already trained on **1.2 million general images** (cats, cars, buildings, etc.) from a dataset called ImageNet. It already knows how to "see." We took that knowledge and specialised it for rice diseases.

The "50" means it has 50 layers deep — each layer learning increasingly complex visual concepts:

```
Early layers  →  edges and colours
Middle layers →  textures and patterns
Deep layers   →  complex shapes and concepts (lesions, discolouration)
```

---

## The Model Architecture

Think of the model as a pipeline with two parts:

### Part 1 — The Backbone (ResNet-50)

This is the "eyes" of the model. It takes the image and converts it into a rich numerical description.

```
Input image (224 × 224 pixels, 3 colour channels)
        ↓
   Initial convolution — detects basic edges
        ↓
   Layer 1 — simple textures
        ↓
   Layer 2 — complex textures
        ↓
   Layer 3 — patterns and shapes
        ↓
   Layer 4 — high-level disease features
        ↓
   Average Pooling — summarises everything
        ↓
   Output: a list of 2048 numbers describing the image
```

Those 2048 numbers are essentially the model's internal "understanding" of what it saw.

### Part 2 — The Classifier Head

This is the "brain" that makes the final decision based on what the eyes saw.

```
2048 numbers in
        ↓
   Dropout (30%) — randomly ignores some numbers during training
                   prevents memorisation, forces generalisation
        ↓
   Linear layer — compresses 2048 → 128 numbers
   + ReLU       — removes negative values, keeps important signals
        ↓
   Dropout (20%) — another regularisation step
        ↓
   Linear layer — compresses 128 → 6 numbers
        ↓
   6 numbers out — one score per disease class
```

The class with the highest score is the prediction.

---

## Training Strategy — Two Phases

Training from scratch with limited data is risky — the model might memorise the training images rather than learn the actual disease patterns. We solve this with a two-phase approach.

### Phase 1 — Teaching the Head (10 epochs)

In this phase we **freeze** the entire backbone — all 50 layers of ResNet-50 are locked, their knowledge untouched. Only the classifier head is allowed to learn.

```
Backbone  →  FROZEN (not learning)
Head      →  LEARNING

Learning rate : 0.001  (relatively fast)
Epochs        : 10
```

Why? The backbone already knows how to see. We just need the head to learn "given this description of an image, which disease is it?" This is fast and safe.

### Phase 2 — Fine-tuning (up to 20 epochs)

Now we unfreeze the deeper layers of the backbone — specifically layer3 and layer4 — and let them adapt to rice specifically.

```
layer1, layer2  →  still FROZEN (basic features, don't touch)
layer3, layer4  →  UNFROZEN (adapting to rice diseases)
Head            →  still LEARNING

Learning rate : 0.0001  (10× slower — careful adjustment)
Epochs        : 20 max
Early stopping: stops automatically if no improvement for 5 epochs
```

Why slower? Because we are adjusting weights that already contain useful knowledge. Too fast and we destroy what ImageNet taught it.

---

## Key Training Decisions

### Label Smoothing
```
Normal loss : model must be 100% confident in the right answer
Smooth loss : model only needs to be ~90% confident
```
This prevents the model from becoming overconfident — important because overconfident models fail badly on images they have never seen before (like your field photos).

### Weighted Random Sampler
The dataset has unequal numbers of images per class:
```
Brown Spot        : ~1600 images
Bacterial Blight  : ~1584 images
Blast             : ~1440 images
Tungro            : ~1308 images
Healthy Rice Leaf : ~653  images   ← half the others
Hispa             : varies
```
Without correction, the model sees disease images far more often than healthy images and becomes biased. The sampler fixes this by making each class appear equally often during training, regardless of how many images it has.

### Augmentation — Making Training Harder on Purpose

We apply random transformations to training images so the model learns to handle real-world variation:

| Augmentation | What it does | Why |
|---|---|---|
| Horizontal flip | Mirrors the image | Diseases look the same from either side |
| Rotation ±15° | Tilts the image | Field photos are never perfectly aligned |
| Colour jitter + hue shift | Changes brightness, contrast, colour | Lighting varies throughout the day |
| Random crop (padding 20px) | Slightly zooms and shifts | Lesions appear at different positions |
| Shadow overlay | Adds random dark polygon | Sunlight creates shadows in the field |
| Solid colour background patch | Covers one side with a flat colour | Simulates blue fabric or wall backgrounds |
| JPEG compression | Degrades image quality slightly | Phone cameras compress images |

Validation and test images receive **none** of these — they are evaluated in their natural state.

---

## How the Model Learns

Each training step follows this loop:

```
1. Feed a batch of 32 images into the model
2. Model produces predictions
3. Compare predictions to correct answers → calculate loss (error)
4. Backpropagation: trace the error back through every layer
5. Adjust weights slightly in the direction that reduces error
6. Repeat for all batches → that is one epoch
```

The scheduler watches the validation loss — if it stops improving, it halves the learning rate automatically, allowing finer adjustments.

---

## Grad-CAM++ — Explaining the Prediction

After prediction, we use Grad-CAM++ to visualise *where* the model was looking when it made its decision.

```
Model makes prediction
        ↓
Gradients flow backwards to layer4
        ↓
We measure: which spatial regions caused the strongest response?
        ↓
Highlight those regions on the original image
        ↓
Red = model focused here strongly
Blue = model ignored this area
```

This is critical for trust — if the model predicts Brown Spot but the heatmap highlights the background instead of the lesion, we know the model is using the wrong features.

The "++" version squares the gradients before averaging, producing tighter, more precise highlights — especially useful for small lesions like Blast and Hispa spots.

---

## Known Limitation

The model performs very well on test images from the same dataset (~100% accuracy). However when tested on true field photos — dense canopy, harsh sunlight, small lesions — confidence drops and some misclassifications occur.

The root cause is **domain gap**: the training images and real field images have different characteristics. The solution is to collect real field images, label them, and add them to the training set. Even 50–100 images per class from the field significantly improves generalisation.