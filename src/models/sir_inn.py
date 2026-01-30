"""
Neural network architecture for SIR-INN models.

This module defines the feedforward neural network used to learn
the latent SIR dynamics within the Physics-Informed Neural Network (PINN)
framework.

The network maps time and epidemiological parameters to normalized
SIR compartment trajectories, enforcing positivity and boundedness
through suitable activation functions.
"""

# ------------------------------------------------------------------

from torch import nn

# ------------------------------------------------------------------

class NN(nn.Module):
    """
    Fully connected neural network for SIR-INN.

    This network is used as a function approximator for the latent
    SIR compartments within the PINN framework.

    The architecture is defined by a list of layer dimensions and
    a chosen activation function. A final Sigmoid activation is
    applied to ensure bounded outputs in [0, 1], consistent with
    normalized population compartments.

    Parameters
    ----------
    dimensions : list[int]
        List specifying the number of neurons in each layer
        (including input and output dimensions).
    activation : torch.nn.Module
        Activation function used between hidden layers
        (e.g. nn.Tanh, nn.ReLU).
    """
    
    def __init__(self, dimensions, activation):
        super(NN, self).__init__()
        layers = []

        # Build hidden layers
        for idx, _ in enumerate(dimensions[:-1]):
            layers += [nn.Linear(dimensions[idx], dimensions[idx+1])]
            layers += [activation()]

        # Remove activation after the last hidden layer
        layers = layers[:-1]

        # Final sigmoid to enforce bounded SIR outputs
        layers += [nn.Sigmoid()]
        
        self.stack = nn.Sequential(*layers)
               
    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing time and model parameters.

        Returns
        -------
        out : torch.Tensor
            Predicted normalized SIR compartments.
        """
        out = self.stack(x)
        return out