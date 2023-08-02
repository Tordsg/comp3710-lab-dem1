import torch
import numpy as np
import matplotlib.pyplot as plt
 
print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]
# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
z = torch.exp(-(x**2+y**2)/2.0)

#creating a 2D sine instead of the Gaussian
zs = torch.sin(x)

#multiply by the gaussian
zx = z * zs

#plot the modulation
plt.imshow(zx.numpy())
plt.tight_layout()
plt.show()




