import torch
import torch as th
import torch.nn as nn
import torchvision.transforms as T
import os
import json
from pathlib import Path
import argparse
from typing import Dict
from PIL import Image
import numpy as np

# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

# Local imports for segmentation model
from .DeepLab import DeepLabV3, DeepLabHead, DeepLabHeadV3Plus
from .utils import IntermediateLayerGetter
from . import resnet


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Part 1: Image Annotation (from data/labeling/local_annotation.py)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class LocalImageAnnotator:
    """
    Generates text annotations for images using a local LLaVA model.
    """
    def __init__(self, model_path: str = "liuhaotian/llava-v1.5-13b"):
        print("Loading LLaVA model, please wait...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            load_8bit=False,
            load_4bit=False,
            device_map="auto"
        )
        print("LLaVA model loaded successfully!")
        
    def process_image_for_llava(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return process_images([image], self.image_processor, self.model.config).to(self.device, dtype=torch.float16)
    
    def annotate_image(self, image_path: str) -> Dict:
        try:
            image_tensor = self.process_image_for_llava(image_path)
            prompt = "Describe this image in JSON format with the following structure: {\"subject\": \"description of main subject\", \"background\": \"description of background\"}. Be concise and specific."
            
            conv = conv_templates["llava_v1"].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], None)
            prompt_with_template = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt_with_template, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[Image.open(image_path).size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=200,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            response = output.split("ASSISTANT:")[-1].strip()
            
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    annotation = json.loads(json_str)
                    if not isinstance(annotation, dict) or "subject" not in annotation or "background" not in annotation:
                        raise ValueError("JSON format incorrect")
                else:
                    raise ValueError("No JSON found")
            except (json.JSONDecodeError, ValueError):
                annotation = {"subject": response, "background": "N/A"}
            
            return {"success": True, "annotation": annotation}
        except Exception as e:
            return {"success": False, "error": str(e)}

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Part 2: Semantic Segmentation (from semantic-segmentation/predict.py)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_color_palette(path):
    """ Loads a color palette from a file. """
    palette = Image.open(path).getpalette()
    return np.array(palette).reshape(-1, 3)

def update_keys(state_dict):
    """ Updates the keys in a model's state dictionary. """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

class SegmentationPredictor:
    """
    Performs semantic segmentation using a DeepLabV3 model.
    """
    def __init__(self, ckpt_path, num_classes=21, output_stride=8, palette_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = deeplabv3plus_resnet101(num_classes=num_classes, output_stride=output_stride, pretrained_backbone=False)
        
        print("Loading segmentation model weights...")
        
        # --- [Compatibility Patch] ---
        # Addresses an issue where loading a PyTorch model saved with an older version of Numpy
        # might fail with "No module named 'numpy._core'". This occurs because older checkpoints
        # have a hardcoded dependency on 'numpy._core.multiarray'. We resolve this by creating
        # an alias in sys.modules, pointing it to the correct module in the current Numpy version.
        import sys
        import numpy
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        # --- [End of Patch] ---

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(update_keys(checkpoint['model_state']))
        self.model.to(self.device)
        self.model.eval()
        print("Segmentation model loaded successfully!")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.palette = get_color_palette(palette_path) if palette_path else None

    def predict(self, image_path, output_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
        
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        
        if self.palette is not None:
            colored_pred = Image.fromarray(pred.astype('uint8')).convert('P')
            colored_pred.putpalette(self.palette)
            colored_pred.save(output_path)
        else:
            # If no palette is provided, save the raw class index map.
            Image.fromarray(pred.astype('uint8')).save(output_path)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Part 3: Main Encoder Class
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class ImageEncoder:
    def __init__(self, annotator_model_path, segmentation_ckpt_path, segmentation_palette_path=None):
        self.annotator = LocalImageAnnotator(model_path=annotator_model_path)
        self.segmenter = SegmentationPredictor(
            ckpt_path=segmentation_ckpt_path,
            palette_path=segmentation_palette_path
        )

    def process_folder(self, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        segmented_output_dir = output_path / 'segmented_images'
        segmented_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_json_path = output_path / 'annotations.json'

        # Check for and load already processed data
        annotations = {}
        if output_json_path.exists():
            with open(output_json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            print(f"Loaded {len(annotations)} records from {output_json_path}.")

        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]

        for i, image_file in enumerate(image_files, 1):
            if image_file.name in annotations:
                print(f"({i}/{len(image_files)}) Skipping already processed file: {image_file.name}")
                continue

            print(f"\n--- ({i}/{len(image_files)}) Processing: {image_file.name} ---")

            # 1. Segmentation
            seg_output_filename = image_file.stem + "_segmented.png"
            seg_output_path = segmented_output_dir / seg_output_filename
            print(f"  -> Generating segmentation map...")
            self.segmenter.predict(str(image_file), str(seg_output_path))
            print(f"  ✓  Segmentation map saved to: {seg_output_path}")

            # 2. Annotation
            print(f"  -> Generating annotation...")
            annotation_result = self.annotator.annotate_image(str(image_file))
            if annotation_result["success"]:
                annotation = annotation_result["annotation"]
                prompt_str = f"{annotation.get('subject', '')}, {annotation.get('background', '')}"
                print(f"  ✓  Annotation successful: {prompt_str[:80]}...")
                
                # 3. Save record - using subject/background structure
                annotations[image_file.name] = {
                    "source": str(seg_output_path.as_posix()),
                    "target": str(image_file.as_posix()),
                    "subject": annotation.get('subject', ''),
                    "background": annotation.get('background', '')
                }
            else:
                print(f"  ✗  Annotation failed: {annotation_result['error']}")
                annotations[image_file.name] = {"error": annotation_result['error']}

            # Save after processing each image
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"\nProcessing complete! Results saved to {output_dir}")

def segmentation_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def load_segmentation_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    model = segmentation_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    return model


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return load_segmentation_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return load_segmentation_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)