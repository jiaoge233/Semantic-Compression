import open_clip
import torch
from torch import nn
import torchvision.transforms as T

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer.
    """
    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cpu", max_length=77,
                 freeze=True, layer="last", output_tokens=False, get_preprocessor=False):
        super().__init__()
        self.output_tokens = output_tokens
        
        # --- FIX: Get image resolution directly from the preprocessing pipeline ---
        model, _, val_preprocess = open_clip.create_model_and_transforms(
            arch, version, device=torch.device('cpu')
        )

        # The val_preprocess is a Compose object. The first transform is usually Resize.
        resize_transform = val_preprocess.transforms[0]
        if not isinstance(resize_transform, T.Resize):
             raise ValueError("Expected the first transform in open_clip preprocessing to be Resize.")
        
        size = resize_transform.size
        # The size can be an int or a (h, w) tuple. We need the int.
        self.input_resolution = size[0] if isinstance(size, (list, tuple)) else size

        del model.transformer
        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        
        print(f"Loaded OpenCLIP vision model {arch} :: {version} with input resolution {self.input_resolution}.")

        if get_preprocessor:
             self.preprocessor = self.preprocess

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        # The self.model is the full CLIP model. Its forward() expects both image and text.
        # To get only the image embeddings, we must call its specific .encode_image() method.
        return self.model.encode_image(image)

    def encode(self, image):
        return self(image)
        
    def encode_image(self, image):
        return self.encode(image)

    @property
    def device(self):
        return self.device 