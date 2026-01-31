# Leaky Tensors: Neuromodulation in Deep Networks

> A Jupyter notebook exploring how neural networks can learn to be robust to weight noise through neuromodulation-inspired training

This project implements a novel approach to training deep neural networks by introducing controlled noise to weights during training, mimicking biological neuromodulation. At each training step, random noise is added to network weights, forcing the model to learn robust representations. The notebook then extends this by training a learnable noise model with finite variance, demonstrating how networks can adapt to and leverage stochastic weight perturbations—similar to how neuromodulators affect biological neural circuits.

## ✨ Features

- **Stochastic Weight Perturbation** — Implements training with random noise added to weights at every epoch, creating a covariance shift that forces the network to learn robust features and generalize better.
- **Learnable Noise Models** — Trains a parametric noise model with finite variance that adapts during training, mimicking biological neuromodulation where the 'noise' itself becomes a learnable signal.
- **Interactive Visualization** — Provides matplotlib-based visualizations showing training dynamics, loss curves, and the effects of weight noise on model performance and robustness.
- **Educational Implementation** — Step-by-step notebook format with 19 cells that progressively build understanding of neuromodulation concepts in deep learning, perfect for learning and experimentation.

## 📦 Installation

### Prerequisites

- Python 3.7+
- Jupyter Lab or Jupyter Notebook
- CUDA-capable GPU (optional, but recommended for faster training)

### Setup

1. pip install torch torchvision numpy matplotlib
   - Installs PyTorch for deep learning, torchvision for datasets, numpy for numerical operations, and matplotlib for visualization
2. pip install jupyter jupyterlab
   - Installs Jupyter environment to run the notebook (skip if already installed or using Google Colab)
3. Clone or download the notebook file to your local machine
   - Get the notebook.ipynb file into your working directory
4. jupyter lab
   - Launches Jupyter Lab in your browser where you can open and run the notebook

## 🚀 Usage

### Run Locally with Jupyter Lab

Launch the notebook on your local machine with Jupyter Lab for full control and customization

```
# Navigate to the project directory
cd path/to/notebook/directory

# Launch Jupyter Lab
jupyter lab

# In the browser, open 'notebook.ipynb' and run cells sequentially
# Use Shift+Enter to execute each cell
```

**Output:**

```
Jupyter Lab opens in browser at http://localhost:8888. Execute cells to see training progress, loss curves, and noise model visualizations.
```

### Run on Google Colab (Cloud)

Upload and run the notebook on Google Colab for free GPU access without local installation

```
# 1. Go to https://colab.research.google.com/
# 2. Click 'File' → 'Upload notebook'
# 3. Upload your notebook.ipynb file
# 4. Run the first cell to install dependencies:
!pip install torch torchvision numpy matplotlib

# 5. Execute remaining cells with Shift+Enter or 'Runtime' → 'Run all'
```

**Output:**

```
Notebook runs in cloud with GPU acceleration. All visualizations and training outputs display inline.
```

### Experiment with Noise Parameters

Modify noise variance and distribution parameters to explore how different neuromodulation strategies affect learning

```
# Look for cells defining noise parameters (typically early in notebook)
# Example modifications:

# Adjust noise variance
noise_std = 0.01  # Try values: 0.001, 0.01, 0.1

# Change noise distribution
noise = torch.randn_like(weight) * noise_std  # Gaussian
# OR
noise = torch.rand_like(weight) * noise_std   # Uniform

# Experiment with learnable variance
learnable_variance = nn.Parameter(torch.ones(layer_size) * 0.01)

# Re-run training cells to see how changes affect convergence
```

**Output:**

```
Different noise configurations produce varying training dynamics, convergence speeds, and final accuracies—demonstrating robustness-accuracy tradeoffs.
```

## 🏗️ Architecture

The notebook is structured as a progressive tutorial with 19 cells that build from basic concepts to advanced neuromodulation models. It starts with standard neural network training, introduces weight noise injection, and culminates in learnable noise models. The architecture follows a typical ML workflow: imports, data loading, model definition, training loop with noise injection, evaluation, and visualization.

### File Structure

```
Notebook Flow:

┌─────────────────────────────────────┐
│  Cell 1-3: Imports & Setup          │
│  - torch, torchvision, numpy, plt   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Cell 4-6: Data Loading             │
│  - MNIST/CIFAR10 dataset            │
│  - DataLoader configuration         │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Cell 7-9: Model Definition         │
│  - Neural network architecture      │
│  - Standard baseline model          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Cell 10-13: Noisy Training         │
│  - Add random noise to weights      │
│  - Train with covariance shift      │
│  - Compare to baseline              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Cell 14-16: Learnable Noise Model  │
│  - Parametric noise with variance   │
│  - Joint optimization               │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│  Cell 17-19: Visualization & Eval   │
│  - Loss curves, accuracy plots      │
│  - Noise variance evolution         │
│  - Robustness analysis              │
└─────────────────────────────────────┘
```

### Files

- **notebook.ipynb** — Main Jupyter notebook containing all code, explanations, and visualizations for the leaky tensor neuromodulation experiments.

### Design Decisions

- Weight noise is added at every training step rather than just during initialization to simulate continuous neuromodulation effects seen in biological systems.
- Learnable noise variance parameters allow the network to adaptively control the amount of stochasticity, similar to how biological systems regulate neuromodulator release.
- The notebook uses a progressive structure (baseline → fixed noise → learnable noise) to clearly demonstrate the impact of each component.
- Visualizations are integrated throughout to provide immediate feedback on how noise affects training dynamics and final performance.
- Simple datasets (MNIST/CIFAR10) are used to keep training times reasonable while still demonstrating the core concepts effectively.

## 🔧 Technical Details

### Dependencies

- **torch** (1.9.0+) — Core deep learning framework for building and training neural networks with automatic differentiation and GPU acceleration.
- **torchvision** (0.10.0+) — Provides standard computer vision datasets (MNIST, CIFAR10) and image transformations for easy data loading.
- **numpy** (1.19.0+) — Numerical computing library for array operations, random number generation, and mathematical functions.
- **matplotlib** (3.3.0+) — Plotting library for creating visualizations of training curves, loss landscapes, and noise distributions.
- **jupyter** — Interactive notebook environment for running code cells, displaying outputs, and combining code with markdown explanations.

### Key Algorithms / Patterns

- Stochastic gradient descent with weight noise injection: At each training step, Gaussian noise is added to weights before forward pass, creating a form of implicit regularization.
- Learnable variance parameters: Noise variance is parameterized and optimized jointly with network weights, allowing adaptive noise scheduling.
- Covariance shift robustness training: By training under continuous weight perturbations, the network learns representations that are stable to parameter variations.
- Neuromodulation-inspired optimization: The learnable noise model mimics biological neuromodulators that adjust neural circuit dynamics in response to task demands.

### Important Notes

- Adding too much noise can prevent convergence—start with small noise_std values (0.001-0.01) and increase gradually to find the sweet spot.
- GPU acceleration is highly recommended for reasonable training times, especially when experimenting with different noise configurations.
- The learnable noise model requires careful initialization of variance parameters to avoid numerical instability (use small positive values).
- This approach increases training time compared to standard methods due to additional noise sampling and parameter updates at each step.
- The biological inspiration is conceptual—real neuromodulation involves complex biochemical processes not fully captured by simple additive noise.

## ❓ Troubleshooting

### Training loss explodes or becomes NaN

**Cause:** Noise variance is too high, causing weight perturbations that destabilize the optimization process.

**Solution:** Reduce the noise_std parameter (try 0.001 or 0.005). If using learnable variance, initialize with smaller values and consider adding variance clipping: variance.data.clamp_(min=1e-6, max=0.1)

### Model doesn't converge or accuracy stays low

**Cause:** Either noise is too high (preventing learning) or learning rate is incompatible with the noise level.

**Solution:** Try reducing both noise variance and learning rate together. Start with noise_std=0.001 and lr=0.0001, then gradually increase. Also ensure you're running enough epochs (20-50).

### CUDA out of memory error

**Cause:** GPU memory is exhausted, possibly due to large batch sizes or model size combined with additional noise parameters.

**Solution:** Reduce batch size in the DataLoader (try 32 or 64 instead of 128). Alternatively, use CPU training by removing .cuda() or .to(device) calls, though this will be slower.

### Notebook cells run slowly or hang

**Cause:** Training on CPU is slow, or the dataset is downloading for the first time, or too many epochs are configured.

**Solution:** Ensure GPU is available and being used (check torch.cuda.is_available()). For first run, wait for dataset download to complete. Reduce number of epochs for faster experimentation (5-10 epochs for testing).

### Visualizations don't display or show errors

**Cause:** Matplotlib backend issues in Jupyter or missing %matplotlib inline magic command.

**Solution:** Add %matplotlib inline at the top of the notebook after imports. If using Jupyter Lab, try %matplotlib widget for interactive plots. Restart kernel if needed.

---

This README was generated to help researchers and students understand neuromodulation-inspired training techniques in deep learning. The notebook serves as both an educational tool and a research prototype for exploring robustness through stochastic weight perturbations. Feel free to experiment with different architectures, datasets, and noise distributions to deepen your understanding of how biological principles can inform machine learning algorithms. The concepts explored here connect to broader topics in robust optimization, Bayesian neural networks, and neuromorphic computing.