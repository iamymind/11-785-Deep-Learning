import torch.nn as nn
from torch.nn import Sequential


class Flatten(nn.Module):
	"""
	Implement a simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
	"""
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self,x):
		n,m = x.size()[0],x.size()[1]
		return x.view(n,m)

def all_cnn_module():
    """
    Create a nn.Sequential model containing all of the layers of the All-CNN-C as specified in the paper.
    https://arxiv.org/pdf/1412.6806.pdf
    Use a AvgPool2d to pool and then your Flatten layer as your final layers.
    You should have a total of exactly 23 layers of types:
    - nn.Dropout
    - nn.Conv2d
    - nn.ReLU
    - nn.AvgPool2d
    - Flatten
    :return: a nn.Sequential model
    """

    model = nn.Sequential(
		nn.Dropout(0.2),
		nn.Conv2d(3, 96, 3, 1, 1),
		nn.ReLU(),
		nn.Conv2d(96, 96, 3, 1, 1),
		nn.ReLU(),
		nn.Conv2d(96, 96, 3, stride=2, padding=1),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Conv2d(96, 192, 3, 1, 1),
		nn.ReLU(),
		nn.Conv2d(192, 192, 3, 1, 1),
		nn.ReLU(),
		nn.Conv2d(192, 192, 3, padding=1, stride=2),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Conv2d(192, 192, 3, padding=0, stride=1),
		nn.ReLU(),
		nn.Conv2d(192, 192, 1, padding=0, stride=1),
		nn.ReLU(),
		nn.Conv2d(192, 10, 1, padding=0, stride=1),
		nn.ReLU(),
		nn.AvgPool2d(6),
		Flatten()
    )
    
    return model
