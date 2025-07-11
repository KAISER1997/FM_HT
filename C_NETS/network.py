# Activation class
import torch
from torch import nn, Tensor
from C_MODELS.utils.extreme_transforms import softplus,_stable_erfcinv,SQRT_2,SQRT_PI

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x

# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)
    

# MLP Tail Param
class MLP_TailParam(nn.Module):
    def __init__(self, time_dim: int = 1, hidden_dim: int = 128,output_dim: int =8):
        super().__init__()

        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.output_dim= output_dim

        self.main = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
            )


    def forward( self,t: Tensor) -> Tensor:
        # sz = x.size()
        # x = x.reshape(-1, self.input_dim)
        # t = t.reshape(-1, self.time_dim).float()

        # t = t.reshape(-1, 1).expand(x.shape[0], 1)
        # h = torch.cat([x, t], dim=1)
        # print("YES",t.shape)
        output = self.main(t.float())

        return output#.reshape(*sz)
    

class MLP_TailParam2(nn.Module):
    def __init__(self, time_dim: int = 1, hidden_dim: int = 128,output_dim: int =8,transform_inp: int=0):
        super().__init__()

        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.output_dim= output_dim
        self.change_input=nn.Linear(output_dim//4,output_dim//4)
        self.transform_inp=transform_inp

        self.main = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim//2),
            Swish(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//4, hidden_dim//8),
            Swish(),
            nn.Linear(hidden_dim//8, output_dim),
            )


    def forward( self,t: Tensor) -> Tensor:
        # sz = x.size()
        # x = x.reshape(-1, self.input_dim)
        # t = t.reshape(-1, self.time_dim).float()

        # t = t.reshape(-1, 1).expand(x.shape[0], 1)
        # h = torch.cat([x, t], dim=1)
        # print("YES",t.shape)
        output = self.main(t.float())

        return output#.reshape(*sz)
    


class MLP2(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim//2),
            # Swish(),
            # nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            Swish(),
            nn.Linear(hidden_dim//4, input_dim),
            )


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)



import torch
import torch.nn as nn
import torch.nn.functional as F
class BigTimeConditionalNet(nn.Module):
    def __init__(self, input_dim=20, time_dim=128, hidden_dim=512):
        super(BigTimeConditionalNet, self).__init__()

        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Fully connected network conditioned on time embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, t):
        """
        x: Tensor of shape (B, 20)
        t: Tensor of shape (B,) - time
        """
        t = t.view(-1, 1)                     # (B, 1)
        t_emb = self.time_mlp(t)              # (B, time_dim)

        x_cat = torch.cat([x, t_emb], dim=-1) # (B, 20 + time_dim)

        out = self.net(x_cat)                 # (B, 20)
        return out

class TimeToVecNet(nn.Module):
    def __init__(self, time_dim=128, hidden_dim=512,output_dim=80 ):
        super(TimeToVecNet, self).__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            # nn.BatchNorm1d(time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
            # nn.BatchNorm1d(time_dim),
            nn.SiLU(),
        )

        # Main network to map time embedding to output vector
        self.net = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, t):
        """
        t: Tensor of shape (B,) - time
        returns: Tensor of shape (B, 80)
        """
        B = t.size(0)
        t = t.view(B, 1)            # (B, 1)
        t_emb = self.time_mlp(t)    # (B, time_dim)
        out = self.net(t_emb)       # (B, 80)
        return out

class FullConnectedScoreModel(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super(FullConnectedScoreModel, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(data_dim+1, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.hidden_layers.append(nn.Sequential(layer, batch_norm))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, data_dim)  # Assuming output is a single value

    def forward(self, x, t):
        x_conc_t = torch.concat([x,t.unsqueeze(1)],axis=1)
        x = F.relu(self.input_batch_norm(self.input_layer(x_conc_t)))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.output_layer(x)
    


class FullConnectedScoreModel_time(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128, n_hidden_layers: int = 2):
        super(FullConnectedScoreModel_time, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(1, hidden_dim)
        self.input_batch_norm = nn.BatchNorm1d(hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_hidden_layers):
            layer = nn.Linear(hidden_dim, hidden_dim)
            batch_norm = nn.BatchNorm1d(hidden_dim)
            self.hidden_layers.append(nn.Sequential(layer, batch_norm))

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, data_dim*4)  # Assuming output is a single value

    def forward(self, t):
        x_conc_t =t
        x = F.relu(self.input_batch_norm(self.input_layer(x_conc_t)))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.output_layer(x)


class MLP_diffusion(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128,output_dims:int=128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dims),
            )


    def forward(self, x: Tensor) -> Tensor:



        output = self.main(x)

        return output

import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import numpy as np

class TTF_layer(nn.Module):
    def __init__(self, dim=2,transform='basic'):  # Fixed constructor name
        super().__init__()      # Fixed super call
        self.lambd_plus = nn.Parameter(torch.randn(dim))
        self.lambd_neg = nn.Parameter(torch.randn(dim))
        self.mu = nn.Parameter(torch.randn(dim))
        self.sigma = nn.Parameter(torch.randn(dim))
        self.transform=transform



    def normalize(self, x):
        return (1 + torch.tanh(x)) / 4

    def forward(self, z):
        if self.transform=='basic':
            sigma = 1e-3 + softplus(self.sigma)
            lambd_plus = self.normalize(self.lambd_plus)+0.1
            lambd_neg = self.normalize(self.lambd_neg)+0.1

            sign = torch.sign(z)
            lambd_s = torch.where(z > 0, lambd_plus, lambd_neg)
            g = torch.erfc(torch.abs(z) / np.sqrt(2)) + 1e-6  # Safe from zero-power issues
            x = (torch.pow(g, -lambd_s) - 1) / lambd_s
            x = sign * x * sigma + self.mu
        elif self.transform=='Recip_deriv_inv':
            sigma = 1e-3 + softplus(self.sigma)
            lambd_plus = softplus(self.lambd_plus)
            lambd_neg = softplus(self.lambd_neg)
            grad=self.dTTFInverse_dz(z,lambd_plus,lambd_neg,self.mu,sigma)
            # print(z.shape,grad.shape,"network")
            x=z/grad




        return x  # Fixed indentation and removed invisible characters
    

    def dTTFInverse_dz(self,x, pos_tail, neg_tail,shift, scale): #aditya wrote this
        s = torch.sign(x - shift)
        
        # Compute λₛ based on sign
        lambda_s = torch.where(s > 0, pos_tail, neg_tail)
        
        # Compute y = λₛ|(x - μ)/σ| + 1
        y = lambda_s * torch.abs((x - shift) / scale) + 1
        
        # Compute y^{-1/λₛ - 1}
        y_pow = torch.pow(y, -1.0/lambda_s - 1)
        
        # Compute erfc^{-1}(y^{-1/λₛ})
        # Note: PyTorch doesn't have direct erfc inverse, so we use inverse of erf and adjust
        # erfc(z) = 1 - erf(z) => erfc^{-1}(w) = erf^{-1}(1 - w)
        w = torch.pow(y, -1.0/lambda_s)
        erfcinv_w =_stable_erfcinv(w, torch.log(w)) #torch.erfinv(1 - w)
        
        # Compute exp(erfcinv_w^2)
        exp_term = torch.exp(torch.square(erfcinv_w))
        
        # Combine all terms
        grad = (1 / scale) * (SQRT_PI/SQRT_2) * y_pow * exp_term
    
        return grad   
 

class MLP_W_ttflayers(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128,transform='basic'):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            # TTF_layer(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            # TTF_layer(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            # TTF_layer(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            # TTF_layer(hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            TTF_layer(input_dim,transform),
            )


    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)
    