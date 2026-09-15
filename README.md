# STL-10 Scrambled Patch Image Reconstruction

A deep learning project that reconstructs a complete **96 × 96 RGB image** from **9 shuffled 28 × 28 image patches** extracted from the STL-10 dataset.

The model must learn both:

- the spatial arrangement of the shuffled patches, and
- the missing visual information between patches caused by cropping.

The solution is implemented entirely with **TensorFlow / Keras**, uses no pretrained models or handcrafted jigsaw-solving algorithm, and contains approximately **2.81 million trainable parameters**.

---

## Overview

Each STL-10 image is divided into a **3 × 3 grid** of 32 × 32 cells. From the center of each cell, a **28 × 28 crop** is extracted.

The nine resulting patches are shuffled before being passed to the network.

**Input**

```text
9 shuffled RGB patches
Shape: (9, 28, 28, 3)
```

**Output**

```text
Reconstructed RGB image
Shape: (96, 96, 3)
```

Unlike a conventional image autoencoder, the network does not receive the original spatial ordering of the image. It therefore needs to infer global structure from the content of the individual patches.

---

## Model Architecture

The reconstruction network combines convolutional feature extraction with attention-based global reasoning.

```text
Shuffled patches
      │
      ▼
Shared CNN Patch Encoder
      │
      ▼
9 Patch Embeddings
      │
      ▼
Slot Positional Embeddings
      │
      ▼
3 × Transformer Blocks
      │
      ▼
Patch Tokens
      │
      ▼
36 Learned Canvas Queries
      │
      ▼
Cross-Attention
      │
      ▼
Canvas Transformer
      │
      ▼
6 × 6 Latent Canvas
      │
      ▼
Convolutional Decoder
      │
      ▼
96 × 96 × 3 Reconstruction
```

### 1. Shared Patch Encoder

A shared CNN independently processes each of the nine patches.

The encoder contains:

- convolutional layers
- batch normalization
- ReLU activations
- downsampling with strided convolutions
- global average pooling
- a dense embedding layer

Each patch is converted into a **192-dimensional feature vector**.

### 2. Patch Transformer

The nine patch embeddings are processed using **three Transformer blocks**.

These attention layers allow every patch to reason about all other patches and learn relationships such as:

- object continuity
- texture similarity
- color consistency
- likely spatial relationships

### 3. Learned Canvas Queries

The model creates **36 trainable query tokens**, representing a latent **6 × 6 image canvas**.

Through multi-head cross-attention, these queries retrieve relevant information from the encoded patch tokens.

This lets the model learn how the unordered patch information should contribute to different regions of the reconstructed image.

### 4. Convolutional Decoder

The 36 canvas tokens are reshaped into a:

```text
6 × 6 × 192
```

feature map.

The decoder progressively upsamples it:

```text
6 × 6
12 × 12
24 × 24
48 × 48
96 × 96
```

Convolutional refinement layers are applied after each upsampling stage.

A final sigmoid-activated convolution produces the reconstructed RGB image.

---

## Dataset

This project uses the **STL-10 unlabeled dataset**.

STL-10 contains 96 × 96 color images and is commonly used for unsupervised and self-supervised learning experiments.

The notebook automatically downloads the official binary dataset if `unlabeled_X.bin` cannot be found locally.

Dataset source:

https://ai.stanford.edu/~acoates/stl10/

### Dataset Split

The 100,000 unlabeled STL-10 images are divided into:

| Split | Images |
|---|---:|
| Training | 80,000 |
| Validation | 10,000 |
| Test | 10,000 |

The original dataset remains stored as `uint8`. Conversion to `float32` and normalization to `[0, 1]` happen only when batches are generated, reducing memory usage.

---

## Patch Generation

For every 96 × 96 image:

1. Divide the image into a 3 × 3 grid.
2. Each grid cell has size 32 × 32.
3. Extract the centered 28 × 28 crop from each cell.
4. Randomly shuffle the nine crops.
5. Use the original image as the reconstruction target.

During training, patch order changes dynamically.

For validation and testing, deterministic permutations are used to keep evaluation reproducible.

---

## Loss Function

The primary evaluation metric is **Mean Absolute Error (MAE)**.

To reduce overly smooth reconstructions, training uses an edge-aware loss:

```text
Loss = Pixel MAE + λ × Gradient MAE
```

where:

```text
λ = 0.15
```

The gradient component compares horizontal and vertical image gradients between the reconstruction and the ground-truth image.

Checkpoint selection is still based on validation MAE.

---

## Results

### Model Size

| Metric | Value |
|---|---:|
| Total parameters | 2,815,683 |
| Trainable parameters | 2,813,187 |
| Non-trainable parameters | 2,496 |

The model remains well below the project's **6 million trainable parameter** constraint.

### Test Performance

| Method | Mean Test MAE | Std. Dev. |
|---|---:|---:|
| Mean-patch baseline | 0.18237 | 0.05612 |
| Proposed model | **0.08528** | **0.03754** |

The neural reconstruction model reduces the mean test MAE by approximately **53%** compared with the baseline.

The best sharpness-fine-tuned model reached a validation MAE of approximately **0.0843**.

---

## Baseline

A simple non-learning baseline is included for comparison.

It:

1. averages all nine shuffled patches,
2. repeats the average patch into a 3 × 3 grid, and
3. resizes the result to 96 × 96.

This baseline achieves a test MAE of approximately:

```text
0.1824
```

compared with:

```text
0.0853
```

for the proposed model.

---

## Training

The model is trained using the Adam optimizer:

```python
keras.optimizers.Adam(learning_rate=3e-4)
```

Training uses:

- model checkpointing
- learning-rate reduction on plateau
- early stopping
- validation MAE for model selection

Default training configuration:

```text
Batch size: 16
Epochs: 30
Initial learning rate: 3e-4
```

For systems with limited GPU memory, the batch size can be reduced to 8.

---

## Sharpness Fine-Tuning

An optional second training stage is included to improve visual sharpness.

The best checkpoint is reloaded and trained for additional epochs using:

```text
Learning rate: 1e-4
Edge-aware loss
```

The notebook saves a separate sharpness-tuned checkpoint:

```text
best_model_sharp.weights.h5
```

---

## Requirements

The project requires:

- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib

Optional:

- Google Colab
- Google Drive
- `gdown` for downloading trained model weights

Install the main dependencies with:

```bash
pip install tensorflow numpy matplotlib
```

If model weights are hosted on Google Drive:

```bash
pip install gdown
```

---

## Running the Project

### Google Colab

The notebook is designed to run directly in Google Colab.

1. Upload the notebook to Colab.
2. Enable a GPU:

```text
Runtime → Change runtime type → GPU
```

3. Run the notebook from top to bottom.
4. The STL-10 dataset will be downloaded automatically if necessary.
5. Training checkpoints will be saved to Google Drive when Drive is mounted.

### Local Environment

Clone or download the repository and install the required packages:

```bash
pip install tensorflow numpy matplotlib
```

Then open the notebook:

```bash
jupyter notebook patches_to_image_spec.ipynb
```

If Google Drive is unavailable, the dataset and outputs are stored locally.

---

## Loading Trained Weights

The notebook supports loading a trained Keras weights file directly.

It also includes optional `gdown` support.

Set:

```python
GDOWN_FILE_ID = "YOUR_FILE_ID_HERE"
```

to the ID of a publicly accessible Google Drive weights file.

The model will then download and load:

```text
best_model.weights.h5
```

---

## Evaluation

The notebook evaluates reconstruction quality using per-image MAE:

```python
MAE = mean(abs(ground_truth - prediction))
```

It reports both:

- mean test MAE
- standard deviation of test MAE

The notebook also visualizes:

1. shuffled input patches
2. ground-truth image
3. reconstructed image
4. absolute reconstruction-error map

---

## Project Constraints

The implementation satisfies the following constraints:

- Neural-network solution only
- Keras / TensorFlow implementation
- No pretrained models
- No external feature extractors
- No handcrafted permutation search
- No explicit jigsaw solver
- Memory-safe batch preprocessing
- Output shape of `(96, 96, 3)`
- Fewer than 6 million trainable parameters

---

## Key Ideas

This project demonstrates how attention mechanisms can be used for **spatial reasoning from unordered visual information**.

Instead of directly predicting the permutation of the patches, the model learns an end-to-end representation in which:

- CNNs extract local visual information,
- self-attention learns relationships among patches,
- learned queries construct a spatial latent representation, and
- a convolutional decoder synthesizes the final image.

This allows the network to solve both **patch arrangement** and **missing-pixel reconstruction** jointly.

## Future Improvements

Possible extensions include:

- stronger patch encoders
- hierarchical or multi-scale attention
- perceptual reconstruction losses
- SSIM-based objectives
- data augmentation
- larger latent canvases
- improved decoder architectures
- explicit visualization of learned attention patterns

---

## Acknowledgements

This project uses the **STL-10 dataset**, introduced by Adam Coates, Andrew Ng, and Honglak Lee.

The implementation was developed using TensorFlow and Keras.

