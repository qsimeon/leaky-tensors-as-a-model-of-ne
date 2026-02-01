# Leaky Tensors: System Architecture & Theory

## 🧠 Core Concept

**Leaky Tensors** is a biologically-inspired training paradigm that injects learnable noise into neural network weights during training. This simulates **neuromodulation** — the process by which neurotransmitters modulate synaptic connections in biological neural circuits.

### The Key Insight

In biological brains, synaptic weights aren't static. Neurotransmitters like dopamine, serotonin, and acetylcholine continuously modulate connection strengths. Networks trained under this constant perturbation become inherently robust.

We model this by:
1. Adding Gaussian noise to weights at **every forward pass** during training
2. Making the noise variance **learnable** — the network learns optimal noise levels
3. Training both the main network and noise model jointly

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LEAKY TENSORS TRAINING LOOP                           │
└─────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐        ┌─────────────────────┐        ┌──────────────────┐
     │    Input     │        │    Noise Model      │        │   Main Network   │
     │   (MNIST)    │        │  (Learnable σ²)     │        │   (LeakyMLP)     │
     └──────┬───────┘        └──────────┬──────────┘        └────────┬─────────┘
            │                           │                            │
            │                           │ Generate Noise             │
            │                           │ ε ~ N(0, σ²)               │
            │                           ▼                            │
            │                    ┌──────────────┐                    │
            │                    │  Noise Dict  │                    │
            │                    │  per layer   │                    │
            │                    └──────┬───────┘                    │
            │                           │                            │
            │                           │ Inject                     │
            │                           ▼                            │
            │              ┌─────────────────────────┐               │
            │              │     WEIGHT + NOISE      │◄──────────────┘
            │              │    W' = W + ε           │
            │              └────────────┬────────────┘
            │                           │
            ▼                           ▼
     ┌────────────────────────────────────────────────────────────────┐
     │                                                                │
     │                        FORWARD PASS                            │
     │                                                                │
     │   Input ──► [LeakyLinear₁] ──► ReLU ──► [LeakyLinear₂] ──►... │
     │              W₁ + ε₁                     W₂ + ε₂               │
     │                                                                │
     └────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  Prediction   │
                              │   (logits)    │
                              └───────┬───────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ CrossEntropy  │◄──── Target Labels
                              │     Loss      │
                              └───────┬───────┘
                                      │
           ┌──────────────────────────┴──────────────────────────┐
           │                    BACKWARD PASS                     │
           │                                                      │
           ▼                                                      ▼
    ┌─────────────────┐                                  ┌────────────────┐
    │  Update Weights │                                  │ Update Noise   │
    │  (Adam, lr=1e-3)│                                  │ Variances      │
    │                 │                                  │ (Adam, lr=1e-4)│
    └─────────────────┘                                  └────────────────┘
```

---

## 📦 Component Details

### 1. LeakyLinear Layer

Custom linear layer that supports noise injection:

```
┌─────────────────────────────────────────────────────┐
│                    LeakyLinear                      │
├─────────────────────────────────────────────────────┤
│  Parameters:                                        │
│    • W: Weight matrix [out_features × in_features]  │
│    • b: Bias vector [out_features]                  │
│    • current_noise: Injected noise tensor           │
├─────────────────────────────────────────────────────┤
│  Forward(x):                                        │
│    if training and noise_injected:                  │
│        return x @ (W + noise)ᵀ + b                  │
│    else:                                            │
│        return x @ Wᵀ + b                            │
└─────────────────────────────────────────────────────┘
```

### 2. Noise Model

Learns optimal noise variance per layer:

```
┌─────────────────────────────────────────────────────┐
│                     NoiseModel                      │
├─────────────────────────────────────────────────────┤
│  Parameters (per layer):                            │
│    • log_σ²: Log-variance (scalar, learnable)       │
│                                                     │
│  Why log-variance?                                  │
│    • Ensures σ² is always positive                  │
│    • More stable gradient flow                      │
│    • Prevents variance collapse to zero             │
├─────────────────────────────────────────────────────┤
│  generate_noise():                                  │
│    σ = exp(0.5 × log_σ²)      # Convert to std     │
│    σ = clamp(σ, 1e-6, 0.1)    # Stability bounds   │
│    ε = randn(shape) × σ       # Sample noise       │
│    return ε                                         │
└─────────────────────────────────────────────────────┘
```

### 3. LeakyMLP Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              LeakyMLP                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Input (784)                                                              │
│       │                                                                    │
│       ▼                                                                    │
│   ┌───────────────┐    ┌───────┐    ┌─────────────┐                       │
│   │ LeakyLinear   │───►│ ReLU  │───►│  Dropout    │                       │
│   │ (784 → 512)   │    │       │    │   (0.2)     │                       │
│   │   + noise₀    │    └───────┘    └──────┬──────┘                       │
│   └───────────────┘                        │                              │
│                                            ▼                              │
│   ┌───────────────┐    ┌───────┐    ┌─────────────┐                       │
│   │ LeakyLinear   │───►│ ReLU  │───►│  Dropout    │                       │
│   │ (512 → 256)   │    │       │    │   (0.2)     │                       │
│   │   + noise₁    │    └───────┘    └──────┬──────┘                       │
│   └───────────────┘                        │                              │
│                                            ▼                              │
│   ┌───────────────┐                                                       │
│   │ LeakyLinear   │───► Output (10 classes)                               │
│   │ (256 → 10)    │                                                       │
│   │   + noise₂    │                                                       │
│   └───────────────┘                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulation

### Standard Forward Pass
$$y = f(Wx + b)$$

### Leaky Forward Pass (Training)
$$y = f((W + \epsilon)x + b), \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

Where:
- $W$ = weight matrix (learned)
- $\epsilon$ = noise matrix (sampled each forward pass)
- $\sigma^2$ = variance (learned by noise model)

### Optimization Objective

We jointly minimize:

$$\mathcal{L}_{total} = \mathbb{E}_{\epsilon}[\mathcal{L}_{CE}(f_\theta(x; W + \epsilon), y)]$$

This expectation forces the network to find weight configurations that are robust across a distribution of perturbations.

---

## 🧬 Biological Inspiration

| Biological System | Leaky Tensors Analog |
|-------------------|----------------------|
| Synaptic Weight | Weight matrix W |
| Neuromodulator (dopamine, etc.) | Noise ε |
| Neuromodulator concentration | Variance σ² |
| Synaptic plasticity | Gradient updates to W |
| Homeostatic regulation | Learned variance adaptation |

### Why This Matters

1. **Robustness**: Networks trained with noise perturbations generalize better to unseen conditions
2. **Biological Plausibility**: More accurately models noisy biological computation
3. **Regularization**: Acts as implicit regularization, similar to dropout but at the weight level
4. **Adversarial Robustness**: Networks become more resistant to adversarial weight perturbations

---

## 📊 Training Flow

```
For each epoch:
    For each batch (x, y):
        │
        ├─► noise_model.generate_noise()     # Sample ε ~ N(0, σ²)
        │
        ├─► model.inject_noise(noise_dict)   # W' = W + ε
        │
        ├─► output = model(x)                # Forward with noisy weights
        │
        ├─► loss = CrossEntropy(output, y)   # Compute loss
        │
        ├─► loss.backward()                  # Backprop through both
        │
        ├─► model_optimizer.step()           # Update W
        │
        ├─► noise_optimizer.step()           # Update σ²
        │
        └─► model.clear_noise()              # Reset for next batch
```

---

## 📈 Expected Behavior

### Noise Variance Evolution
- **Early training**: Network may benefit from higher noise (exploration)
- **Later training**: Noise variance typically decreases (exploitation)
- **Per-layer differences**: Different layers may learn different optimal variances

### Robustness Characteristics
- Model trained with neuromodulation should degrade gracefully under inference-time noise
- Standard models (no noise training) collapse quickly when weights are perturbed

---

## 🔗 Connections to Related Work

| Technique | Relationship to Leaky Tensors |
|-----------|------------------------------|
| **Dropout** | Noise on activations vs. noise on weights |
| **Weight Decay** | Static regularization vs. dynamic perturbation |
| **Bayesian Neural Networks** | Full posterior vs. learned noise variance |
| **Shake-Shake Regularization** | Similar concept for residual branches |
| **Noisy Networks (NoisyNet)** | Exploration in RL via weight noise |

---

## 🚀 Usage

```python
# Create leaky model
model = create_model('mlp', input_dim=784, hidden_dims=[512, 256], output_dim=10)

# Create noise model with proper layer shapes
layer_shapes = {f'layer_{i}': l.weight.shape for i, l in enumerate(model.get_leaky_layers())}
noise_model = NoiseModel(layer_shapes)

# Training loop injects noise at each step
noise_dict = noise_model.generate_noise()
model.inject_noise(noise_dict)
output = model(x)
# ... compute loss and backprop ...
model.clear_noise()
```

---

*This architecture document accompanies the Leaky Tensors notebook demonstrating neuromodulation in deep networks.*
