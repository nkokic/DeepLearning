import torch.nn as nn
import torch.nn.parallel
import matplotlib.pyplot as plt
from LV3_network import Generator, Discriminator
from LV3_dataloaders import batch_size, dataloader, dataset
from LV3_network import latentSize
import numpy as np
import torch_fidelity

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modelG = Generator().to(device)
modelG.load_state_dict(torch.load("LV3/models/generator.pth", weights_only=True))
modelD = Discriminator().to(device)
modelD.load_state_dict(torch.load("LV3/models/discriminator.pth", weights_only=True))

print("")
print("Generator number of parameters: " + str(sum(p.numel() for p in modelG.parameters())))
print("Discriminator number of parameters: " + str(sum(p.numel() for p in modelD.parameters())))

from torch_fidelity import calculate_metrics
import torch
from torchvision.utils import save_image
import os

fake_path = "LV3/fake_imgs"
real_path = "LV3/celeba/img_align_celeba"

num_fake_images =  10000  
n_batches = int(num_fake_images / batch_size)


# iters= 0
# for i, data in enumerate(dataloader, 0):
#     fixed_noise = torch.randn(64, latentSize, 1, 1, device=device)
#     fake = modelG(fixed_noise).detach().cpu()
#     flattened_outmap = fake.view(fake.shape[0], -1, 1, 1) # Use 1's to preserve the number of dimensions for broadcasting later, as explained
#     outmap_min, _ = torch.min(flattened_outmap, dim=1, keepdim=True)
#     outmap_max, _ = torch.max(flattened_outmap, dim=1, keepdim=True)
#     fake = (fake - outmap_min) / (outmap_max - outmap_min)
#     for f in fake:
#         f = np.transpose(f, (1, 2 ,0))
#         max_img = len(dataloader) * batch_size
#         print(f"{iters}/{max_img}")
#         plt.imsave(f"{fake_path}/{iters}.jpg", f)
#         iters += 1

# print(f"Generated {num_fake_images} fake images in batches of {batch_size}")

# ---------------------------
# COMPUTE FID
# ---------------------------

if __name__ == '__main__':
    wrapped_generator = torch_fidelity.GenerativeModelModuleWrapper(modelG, 100, 'normal', 0)

    print(wrapped_generator)

    metrics_dict = torch_fidelity.calculate_metrics(
        input1=fake_path,
        input2=dataset,
        batch_size=batch_size,
        cuda=True,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
    )

    print(metrics_dict)