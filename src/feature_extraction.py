# src/feature_extraction.py

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device

        # Load ResNet18 with pretrained ImageNet weights
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()  # type: ignore  
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard transform with data augmentation for robustness
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Separate transform for ball (smaller objects)
        self.ball_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),  # Square aspect ratio for ball
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Clean transform without augmentation for stable tracking
        self.clean_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image, object_type='player', stable=True):
        with torch.no_grad():
            # Use appropriate transform based on object type and stability requirement
            if object_type == 'ball':
                if stable:
                    img_tensor = self.ball_transform(image).unsqueeze(0).to(self.device)
                else:
                    img_tensor = self.ball_transform(image).unsqueeze(0).to(self.device)
            else:
                if stable:
                    img_tensor = self.clean_transform(image).unsqueeze(0).to(self.device)
                else:
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            embedding = self.model(img_tensor)
        return embedding.cpu().numpy().flatten()
