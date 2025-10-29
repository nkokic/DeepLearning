import torch.optim as optim
from tqdm import tqdm
import numpy as np
import progressbar
from LV2_UNet import *

# IoU Metric
def iou_score(output, target, num_classes=4):
    # Convert output to one-hot encoding
    output = torch.argmax(output, dim=1)  # Get predicted class for each pixel
    iou = []
    for i in range(num_classes):
        intersection = torch.sum((output == i) & (target == i)).item()
        union = torch.sum((output == i) | (target == i)).item()
        iou.append(intersection / (union + 1e-6))  # Avoid division by zero
    return np.mean(iou)  # Return mean IoU over all classes

# Define training function
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            # Move data to the same device as the model
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        val_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                val_iou += iou_score(outputs, masks)

        print(f"Epoch {epoch+1}/{epochs}, Validation IoU: {val_iou/len(val_loader)}")

# Instantiate model
model = UNet(in_channels=1, out_channels=4).to(device)

# Train the model
train_model(model, train_loader, val_loader, epochs=10, lr=1e-3)