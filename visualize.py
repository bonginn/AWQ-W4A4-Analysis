import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load .pth file
activations_dict = torch.load('./act_scales/input_feat.pt', map_location='cpu', weights_only=True)

# Currently the best to visualize outlier: model.layers.N.self_attn.q_proj
layer_name = 'model.layers.31.self_attn.q_proj'
layer_acts = activations_dict[layer_name]

acts = torch.stack(layer_acts, dim=0).cpu().numpy()  # shape: [T, C]

# Build Coordinate grid
tokens, channels = acts.shape
T, C = np.meshgrid(np.arange(tokens), np.arange(channels), indexing="ij")

# Drawing
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(C, T, acts, cmap='coolwarm', linewidth=0, antialiased=False)

ax.set_xlabel('Channel')
ax.set_ylabel('Token')
ax.set_zlabel('Absolute Value')
plt.title(f'Activation of {layer_name}')

plt.show()