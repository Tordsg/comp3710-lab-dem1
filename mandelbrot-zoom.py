import torch
import numpy as np
import matplotlib.pyplot as plt
 
print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

#Creating a higher resolution zoomed in image
Y, X = np.mgrid[0.3:0.4:0.0001, -0.8:-0.55:0.0001]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y)

zs = torch.zeros_like(z)

zs = z

ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

#Mandelbrot Set
for i in range(200):
    #Compute the new values of z: z^2 + x
    zs_ = zs*zs + z
    #Have we diverged with this new value? 
    not_diverged = torch.abs(zs_) < 4.0 
    #Update variables to compute
    ns += not_diverged.type(torch.FloatTensor) 
    zs = zs_
    
#plot
fig = plt.figure(figsize=(16,10))
def processFractal(a):
    """Display an array of iteration counts as a
            colorful picture of a fractal."""
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a==a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.numpy()))
plt.tight_layout(pad=0)
plt.show()


