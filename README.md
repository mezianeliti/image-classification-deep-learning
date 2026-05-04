# Image Classification with Deep Learning

MLP-based image classification on the **Fashion MNIST** dataset using Keras + TensorFlow.

## Dataset

[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) — 70,000 grayscale images (28×28px) across 10 clothing categories.

| Split | Size |
|-------|------|
| Train | 55,000 |
| Val   | 5,000 |
| Test  | 10,000 |

## Model Architecture

```
Input(28×28) → Flatten → Dense(300, ReLU) → Dense(100, ReLU) → Dense(10, Softmax)
```

- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Weight init: He Normal

## Key Experiments

- Effect of input normalization (÷255 vs raw vs ÷25000)
- ReLU vs Sigmoid activations
- Impact of weight initialization variance

## Stack

- Python · Keras 3 · TensorFlow · NumPy · Matplotlib

## Usage

```bash
jupyter notebook image_classification.ipynb
```
