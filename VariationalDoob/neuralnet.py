"""
A module to approximate functions with neural networks.
"""

import torch
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final = False, activation_fn = torch.nn.LeakyReLU()):
        """
        Parameters
        ----------    
        input_dim : int specifying input_dim of input 

        layer_widths : list specifying width of each layer 
            (len is the number of layers, and last element is the output input_dim)

        activate_final : bool specifying if activation function is applied in the final layer

        activation_fn : activation function for each layer        
        """
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x

class V_Network(torch.nn.Module):

    def __init__(self, dimension_state, num_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        num_obs : number of observations
        config : dict containing      
            layers : list specifying width of each layer 
        """
        super().__init__()
        input_dimension = dimension_state        
        layers = config['layers']
        self.net = torch.nn.ModuleList([MLP(input_dimension, 
                                        layer_widths = layers + [1]) for t in range(num_obs - 1)])

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : observation index (int)
        
        x : state (N, d)
                        
        Returns
        -------    
        out :  output (N)
        """
        
        out = torch.squeeze(self.net[t-1](x)) # size (N)
            
        return out


class MLP_Z_Network(torch.nn.Module):
    def __init__(self, dimension_state, layers):
        super(MLP_Z_Network, self).__init__()
        self.net = MLP(dimension_state + 1, layer_widths = layers + [dimension_state])

    def forward(self, s, x):        
        """
        Parameters
        ----------
        s : time step (N, 1)

        x : state (N, d)
                        
        Returns
        -------    
        out :  output (N, d)
        """
        N = x.shape[0]
        if len(s.shape) == 0:
            s_ = s.repeat((N, 1))
        else:
            s_ = s
        
        h = torch.cat([s_, x], -1) # size (N, d+1)
        out = self.net(h) # size (N, d)
     
        return out

class Z_Network(torch.nn.Module):

    def __init__(self, dimension_state, num_obs, config):
        """
        Parameters
        ----------
        dimension_state : int specifying state dimension
        num_obs : number of observations
        config : dict containing      
            layers : list specifying width of each layer 
        """
        super().__init__()
        layers = config['layers']        
        self.net = torch.nn.ModuleList([MLP_Z_Network(dimension_state, layers) for t in range(num_obs)])

    def forward(self, t, s, x):
        """
        Parameters
        ----------
        t : observation index (int)

        s : time step (tensor)

        x : state (N, d)
                        
        Returns
        -------    
        out :  output (N, d)
        """
        
        out = self.net[t](s, x) # size (N, d)

        return out
