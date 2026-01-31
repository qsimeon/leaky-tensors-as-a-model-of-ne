"""
Neural network model with leaky tensor support for neuromodulation.

This module implements a neural network that supports adding learned noise
to weights during training, modeling neuromodulation in biological systems.
The network must learn to be robust to this covariance shift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import numpy as np


class LeakyLinear(nn.Module):
    """
    Linear layer with support for additive noise injection during training.
    
    The noise is added to weights at each forward pass during training,
    forcing the network to learn robust representations.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(LeakyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Noise will be injected from external noise model
        self.current_noise = None
        
    def inject_noise(self, noise: torch.Tensor):
        """Inject noise tensor to be added to weights."""
        self.current_noise = noise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noise injection."""
        if self.training and self.current_noise is not None:
            noisy_weight = self.weight + self.current_noise
            return F.linear(x, noisy_weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)
    
    def clear_noise(self):
        """Clear injected noise."""
        self.current_noise = None


class LeakyConv2d(nn.Module):
    """
    Convolutional layer with support for additive noise injection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super(LeakyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Standard weight and bias parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        self.current_noise = None
        
    def inject_noise(self, noise: torch.Tensor):
        """Inject noise tensor to be added to weights."""
        self.current_noise = noise
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noise injection."""
        if self.training and self.current_noise is not None:
            noisy_weight = self.weight + self.current_noise
            return F.conv2d(x, noisy_weight, self.bias, self.stride, self.padding)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
    
    def clear_noise(self):
        """Clear injected noise."""
        self.current_noise = None


class LeakyMLP(nn.Module):
    """
    Multi-layer perceptron with leaky tensor support.
    
    This network uses LeakyLinear layers that can have noise injected
    into their weights during training.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float = 0.1):
        super(LeakyMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(LeakyLinear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(LeakyLinear(prev_dim, output_dim))
        
        self.layers = nn.ModuleList(layers)
        
    def get_leaky_layers(self) -> List[nn.Module]:
        """Return all leaky layers that can have noise injected."""
        return [layer for layer in self.layers if isinstance(layer, LeakyLinear)]
    
    def inject_noise(self, noise_dict: Dict[str, torch.Tensor]):
        """
        Inject noise into all leaky layers.
        
        Args:
            noise_dict: Dictionary mapping layer names to noise tensors
        """
        leaky_layers = self.get_leaky_layers()
        for idx, layer in enumerate(leaky_layers):
            key = f'layer_{idx}'
            if key in noise_dict:
                layer.inject_noise(noise_dict[key])
    
    def clear_noise(self):
        """Clear noise from all leaky layers."""
        for layer in self.get_leaky_layers():
            layer.clear_noise()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x


class LeakyCNN(nn.Module):
    """
    Convolutional neural network with leaky tensor support.
    
    Suitable for image classification tasks with noise injection.
    """
    
    def __init__(self, input_channels: int, num_classes: int, 
                 conv_channels: List[int] = [32, 64, 128],
                 fc_dims: List[int] = [256]):
        super(LeakyCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        prev_channels = input_channels
        
        for channels in conv_channels:
            conv_layers.append(LeakyConv2d(prev_channels, channels, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            prev_channels = channels
        
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Fully connected layers
        fc_layers = []
        # Assuming input size of 32x32, after 3 pooling layers: 4x4
        prev_dim = prev_channels * 4 * 4
        
        for fc_dim in fc_dims:
            fc_layers.append(LeakyLinear(prev_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.5))
            prev_dim = fc_dim
        
        fc_layers.append(LeakyLinear(prev_dim, num_classes))
        
        self.fc_layers = nn.ModuleList(fc_layers)
        
    def get_leaky_layers(self) -> List[nn.Module]:
        """Return all leaky layers that can have noise injected."""
        leaky = []
        for layer in self.conv_layers:
            if isinstance(layer, LeakyConv2d):
                leaky.append(layer)
        for layer in self.fc_layers:
            if isinstance(layer, LeakyLinear):
                leaky.append(layer)
        return leaky
    
    def inject_noise(self, noise_dict: Dict[str, torch.Tensor]):
        """Inject noise into all leaky layers."""
        leaky_layers = self.get_leaky_layers()
        for idx, layer in enumerate(leaky_layers):
            key = f'layer_{idx}'
            if key in noise_dict:
                layer.inject_noise(noise_dict[key])
    
    def clear_noise(self):
        """Clear noise from all leaky layers."""
        for layer in self.get_leaky_layers():
            layer.clear_noise()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x


def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create neural models.
    
    Args:
        model_type: Type of model ('mlp' or 'cnn')
        **kwargs: Model-specific arguments
        
    Returns:
        Neural network model with leaky tensor support
    """
    if model_type == 'mlp':
        return LeakyMLP(**kwargs)
    elif model_type == 'cnn':
        return LeakyCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

=