# src/transforms.py

from torchvision import transforms

# ImageNet mean and std, widely used for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Transform for training set: includes data augmentation
train_transform = transforms.Compose([
    # Resize the shorter side to 256, then center crop to 224x224
    transforms.Resize((256, 256)),
    # Random crop can increase data diversity
    transforms.RandomCrop((224, 224)),
    # Random horizontal flip is reasonable for many anime images
    transforms.RandomHorizontalFlip(p=0.5),
    # Slight color jitter to improve robustness (especially for hair color)
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.02,
    ),
    # Convert PIL image (0-255) to FloatTensor (0-1)
    transforms.ToTensor(),
    # Normalize using ImageNet statistics
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Transform for validation / test: no random augmentation, only deterministic ops
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
