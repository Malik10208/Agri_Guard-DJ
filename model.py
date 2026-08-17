import torch
from torchvision import datasets, transforms

# These numbers are standard for pretrained models
# Don't change them
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def get_loaders(data_dir, batch_size=16):
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_ds = datasets.ImageFolder(data_dir+'/train', train_tf)
    val_ds   = datasets.ImageFolder(data_dir+'/val',   val_tf)
    train_ld = torch.utils.data.DataLoader(train_ds,
                   batch_size=batch_size, shuffle=True)
    val_ld   = torch.utils.data.DataLoader(val_ds,
                   batch_size=batch_size)
    return train_ld, val_ld, train_ds.classes

import torchvision.models as models
import torch.nn as nn

def get_model(num_classes=2):
    model = models.mobilenet_v2(pretrained=True)

    # Freeze all layers — don't touch what MobileNet knows
    for param in model.parameters():
        param.requires_grad = False

    # Only replace and train the last layer
    model.classifier[1] = nn.Linear(
        model.last_channel, num_classes
    )
    return model