import torch
from torch import nn
import numpy as np

class IKNet(nn.Module):
    def __init__(self):
        super(IKNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 8, True),
            nn.ReLU(),
            nn.Linear(8, 2, False)
        )

    def forward(self, x):
        y = self.linear_relu_stack(fk_input(x))
        return y

def fk_input(x):
    input = torch.cat((torch.sin(x), torch.cos(x), torch.sin(x + torch.roll(x, 1, dims=1)),
                       torch.cos(x + torch.roll(x, 1, dims=1))), dim=1)
    return input

def get_output(y):
    output = y
    return output

def abs_angle_error(q1, q2):
    pi = torch.tensor(np.pi)
    if q1 > 0:
        q1p = torch.fmod(q1, 2 * pi)
        q1n = torch.abs(torch.fmod(q1, 2 * pi) - 2 * pi)
    else:
        q1p = torch.fmod(q1, 2 * pi) + 2 * pi
        q1n = torch.abs(torch.fmod(q1, 2 * pi))

    if q2 > 0:
        q2p = torch.fmod(q2, 2 * pi)
        q2n = torch.abs(torch.fmod(q2, 2 * pi) - 2 * pi)
    else:
        q2p = torch.fmod(q2, 2 * pi) + 2 * pi
        q2n = torch.abs(torch.fmod(q2, 2 * pi))

    print(q1p, q1n)
    print(q2p, q2n)

    return torch.minimum(torch.abs(q1p - q2p), torch.abs(q1n - q2n))

def custom_loss(yhat, y):
    # dy = abs_angle_error(yhat, y)
    # return torch.mean(dy**2)
    return torch.mean((y-yhat)**2)
