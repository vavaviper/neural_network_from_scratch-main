# Neural Network from Scratch

This project demonstrates how to build and train a fully-connected neural network from scratch using only NumPy. The network is designed to classify handwritten digits from the MNIST dataset, achieving high accuracy without relying on deep learning frameworks for the core model logic.

## Features

- **Custom Neural Network Implementation:** No PyTorch or TensorFlow for the model itself—just NumPy.
- **He Initialization:** For improved convergence in deep networks.
- **ReLU Activation:** Used in hidden layers for non-linearity.
- **Softmax Output:** For multi-class classification.
- **Cross-Entropy Loss:** Standard for classification tasks.
- **Mini-batch Gradient Descent:** For efficient training.
- **One-hot Encoding:** For target labels.
- **Training Visualization:** Loss curve plotted over epochs.

## Results

- **Test Accuracy:** Achieves ~96.67% accuracy on the MNIST test set after 20 epochs.
- **Loss Curve:** Training loss decreases steadily, indicating effective learning.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- TensorFlow (only for loading the MNIST dataset)

Install dependencies with:

```bash
pip install numpy matplotlib tensorflow
```

### Running the Code

Simply run:

```bash
python nnfs.py
```

This will:

- Download and preprocess the MNIST dataset.
- Train a neural network with architecture `[784, 128, 64, 10]`.
- Print training loss per epoch and final test accuracy.
- Display a plot of the training loss over epochs.

## Neural Network Architecture

- **Input Layer:** 784 units (28x28 pixels, flattened)
- **Hidden Layer 1:** 128 units, ReLU activation
- **Hidden Layer 2:** 64 units, ReLU activation
- **Output Layer:** 10 units, Softmax activation

## File Structure

- `nnfs.py` — Main script containing the neural network implementation and training loop.
- `README.md` — Project documentation.

## Example Output

```
Epoch 1/20 - Loss: 0.4592
...
Epoch 20/20 - Loss: 0.0509

Test Accuracy: 96.67%
```

## License

This project is for educational purposes and is provided under the MIT License.
