import torch
import torch.nn as nn
import clip
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet101, ResNet101_Weights

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms.functional import normalize

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPExtractor(nn.Module):
    def __init__(self, model_type="ViT-B/32", device=device, download_dir=None):
        super(CLIPExtractor, self).__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            model_type, device=device, download_root=download_dir
        )

        # Text prompt for analysis
        self.prompt = [
            "Analyze the image for indicators of authenticity or manipulation, "
            "focusing on natural textures, lighting, facial feature consistency, "
            "background realism, and possible artifacts such as mismatched reflections, "
            "edge transitions, or unnatural elements."
        ]
        self.text_tokens = clip.tokenize(self.prompt).to(device)

    def forward(self, images):
        """
        Forward pass to extract features.

        Args:
            images: A batch of preprocessed images [batch_size, 3, 224, 224].

        Returns:
            image_features: Image embeddings for the batch [batch_size, feature_dim].
            text_features: Text embeddings for the prompt [1, feature_dim].
            similarity: Optional similarity scores [batch_size].
        """
        images = images.to(self.device)

        # Extract embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(
                images
            )  # [batch_size, feature_dim]
            text_features = self.model.encode_text(self.text_tokens)  # [1, feature_dim]

        # Normalize the embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_features = text_features.expand(images.size(0), -1)
        return image_features, text_features


class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.model = resnet101(weights=ResNet101_Weights.DEFAULT)

        self.model.fc = nn.Linear(2048, 512)

    def forward(self, x):
        return self.model(x)


class CustomClassifier(nn.Module):
    def __init__(self):
        super(CustomClassifier, self).__init__()
        # Define MLP layers
        input_size = (
            512 * 3
        )  # Combined size of backbone img, clip img and clip text embeddings
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.4)

    def forward(self, bb_embed, img_emb, text_emb):
        # Concatenate the embeddings along the feature dimension
        x = torch.cat(
            [bb_embed, img_emb, text_emb], dim=1
        )  # Concatenate along the feature axis
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


class CustomModel(nn.Module):
    def __init__(self, backbone_model):
        super(CustomModel, self).__init__()
        self.backbone = backbone_model.to(device)
        self.extractor = CLIPExtractor().to(device)
        self.classifier = CustomClassifier().to(device)
        self.clip_preprocess = self.extractor.preprocess

    def forward(self, x):
        img_emb, text_emb = self.extractor(x)  # get text embeddings based on img
        bb_embed = self.backbone(x)  # get image embeddings
        x = self.classifier(img_emb, text_emb, bb_embed)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM with a model and the target convolutional layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # If class_idx is None, take the class with the max score
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[:, class_idx].backward()

        # Compute Grad-CAM heatmap
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))  # Global Average Pooling on gradients
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_tensor.size(3), input_tensor.size(2)))  # Resize to input size
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def overlay_heatmap(self, heatmap, image, alpha=0.5):
        """
        Overlay the Grad-CAM heatmap onto the original image.

        Args:
            heatmap: Grad-CAM heatmap (2D array).
            image: Original PIL image.
            alpha: Transparency level for the heatmap overlay.

        Returns:
            overlay: Heatmap blended with the original image.
        """
        # Ensure heatmap and image have the same dimensions
        image = np.array(image)  # Convert PIL image to NumPy array
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # Resize heatmap

        # Normalize heatmap to [0, 255]
        heatmap = np.uint8(255 * heatmap)

        # Convert the heatmap to a color map
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Ensure the original image is in BGR format for OpenCV
        if image.shape[-1] == 3:  # If RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Overlay heatmap on the original image
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

        # Convert back to RGB for display
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay

