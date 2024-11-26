import torch
from tqdm import tqdm
from ml.utils import set_seed
from ml.model import ResnetModel, CustomModel, GradCAM

class InferenceWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 64
        self.seed = 42
        self.lr = 1e-4
        self.num_epochs = 20
        set_seed(self.seed)

    def load_model(self):
        # Load the model
        self.model = CustomModel(backbone_model=ResnetModel()).to(self.device)
        # Load the trained model
        self.model.load_state_dict(torch.load("../weights/90_5_clip_extractor.pth"))
        self.model.load_data(self.batch_size)
        self.model = self.model.to(self.device)

        # Initialize Grad-CAM
        self.grad_cam = GradCAM(
            self.model, target_layer=self.model.backbone.model.layer4
        )

    def predict_and_explain(self, image):
        """
        Predict and generate Grad-CAM heatmap.
        """
        # Preprocess the PIL image
        input_tensor = self.model.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor).squeeze().item()

        # Generate Grad-CAM heatmap
        heatmap = self.grad_cam.generate_heatmap(input_tensor)

        # Overlay heatmap on the image
        overlay = self.grad_cam.overlay_heatmap(heatmap, image)

        return prediction, overlay
