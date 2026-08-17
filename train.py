import torch
import torch.nn as nn
from model import get_loaders, get_model

DATA_DIR = './data'
EPOCHS   = 5
LR       = 0.001

train_loader, val_loader, classes = get_loaders(DATA_DIR)
print("Classes found:", classes)

device = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')
model  = get_model(num_classes=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    print(f"Epoch {epoch+1}/{EPOCHS} "
          f"| Accuracy: {100*correct/total:.1f}%")

torch.save(model.state_dict(), 'crop_model.pth')
print("Model saved to crop_model.pth")