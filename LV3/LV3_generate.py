import torch.nn as nn
import torch.nn.parallel
import matplotlib.pyplot as plt
from LV3_network import Generator
from LV3_network import latentSize
import numpy as np
import time


start_time = time.time()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelG = Generator().to(device)
modelG.load_state_dict(torch.load("LV3/models/generator.pth", weights_only=True))

load_time = time.time()

fixed_noise = torch.randn(1, latentSize, 1, 1, device=device)

fake = modelG(fixed_noise).detach().cpu()

generate_time = time.time()

flattened_outmap = fake.view(fake.shape[0], -1, 1, 1)
outmap_min, _ = torch.min(flattened_outmap, dim=1, keepdim=True)
outmap_max, _ = torch.max(flattened_outmap, dim=1, keepdim=True)
fake = (fake - outmap_min) / (outmap_max - outmap_min)

fake = np.transpose(fake[0], (1, 2 ,0))

print(f"Time to load: {load_time - start_time} s")
print(f"Time to generate: {generate_time - load_time} s")

plt.imshow(fake)
plt.show()