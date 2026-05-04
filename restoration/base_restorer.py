from abc import ABC, abstractmethod
import torch
import logging

logger = logging.getLogger(__name__)

class BaseRestorer(ABC):
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    @abstractmethod
    def _load_model(self):
        """Load the model from the given path."""
        pass

    @abstractmethod
    def restore(self, image, **kwargs):
        """Restore the image. Return the restored image."""
        pass

    def _to_tensor(self, image):
        """Convert PIL or numpy image to tensor."""
        import torchvision.transforms as transforms
        if isinstance(image, torch.Tensor):
            return image.to(self.device)
        transform = transforms.ToTensor()
        return transform(image).unsqueeze(0).to(self.device)

    def _to_pil(self, tensor):
        """Convert tensor to PIL image."""
        import torchvision.transforms as transforms
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        transform = transforms.ToPILImage()
        return transform(tensor.cpu())