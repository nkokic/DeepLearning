from LV2_train import *

# Evaluate on Test Set
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()
test_iou = 0.0
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        test_iou += iou_score(outputs, masks)

print(f"Test IoU: {test_iou/len(test_loader)}")