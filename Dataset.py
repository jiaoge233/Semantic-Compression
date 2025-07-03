# my_controlnet_dataset.py
import json
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image 

class MyControlNetDataset(Dataset):
    def __init__(self, json_file_path, image_resolution=512):
        """
        Initializes the dataset object.
        Args:
            json_file_path (str): Path to the JSON file containing the dataset description.
                                 The JSON file should be a list of dictionaries, where each dictionary contains:
                                 - "source": Path to the control condition image (sketch).
                                 - "target": Path to the target image.
                                 - "prompt": The text prompt string.
            image_resolution (int): The resolution to which images will be resized (defaults to 512x512).
        """
        super().__init__()
        self.data = []
        with open(json_file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
            
        self.image_resolution = image_resolution
        print(f"Dataset initialized, loaded {len(self.data)} items.")

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and returns a training sample by index.
        Args:
            idx (int): The index of the sample.
        Returns:
            dict: A dictionary containing the processed target image, text prompt, and control condition image (sketch).
                  Keys are: 'jpg' (target image), 'txt' (text prompt), 'hint' (control image/sketch)
        """
        item = self.data[idx]

        source_path = item['source'] # Control image path
        target_path = item['target'] # Target image path
        prompt = item['prompt']      # Text prompt

        try:
            source_image_pil = Image.open(source_path).convert('RGB')
            target_image_pil = Image.open(target_path).convert('RGB')
        except Exception as e:
            print(f"Error: Could not load image {source_path} or {target_path}. Error: {e}")
            raise e

        # Preprocess the target image
        target_image_np = np.array(target_image_pil).astype(np.uint8)
        target_image_resized = cv2.resize(target_image_np, (self.image_resolution, self.image_resolution), interpolation=cv2.INTER_AREA)
        target_image_normalized = (target_image_resized.astype(np.float32) / 127.5) - 1.0 # Normalize to [-1, 1]

        # Preprocess the control image (sketch)
        # For sketches, it is usually a single-channel grayscale image, or it can also be an RGB image.
        # The ControlNet Scribble model expects an input similar to boundary lines, usually black and white.
        # You need to adjust this part according to your sketch data and the requirements of the Scribble model.
        # The following is a general processing method, assuming the sketch is also RGB (if not, you need to adjust .convert())
        source_image_np = np.array(source_image_pil).astype(np.uint8)
        source_image_resized = cv2.resize(source_image_np, (self.image_resolution, self.image_resolution), interpolation=cv2.INTER_AREA)
        
        # For Scribble-type control, the input is usually a single-channel binarized image or grayscale image.
        # If your sketch is colored, you may need to convert it to grayscale first.
        # If it is already in a suitable format (e.g., black and white line art), not much extra processing may be needed.
        # Here we assume that the Scribble model can process RGB images normalized to [0, 1] or [-1, 1] (like HED).
        # Or more commonly, a single-channel grayscale image normalized to [0,1].
        # For simplicity, we first process as RGB and normalize to [-1,1]. This can be adjusted later according to the specific input requirements of the Scribble model.
        if source_image_resized.ndim == 2: # If it's a grayscale image
            source_image_resized = cv2.cvtColor(source_image_resized, cv2.COLOR_GRAY2RGB) # Convert to 3 channels
        
        source_image_normalized = (source_image_resized.astype(np.float32) / 127.5) - 1.0 # Normalize to [-1, 1]
        # Or for a single-channel control map in [0,1]:
        # source_image_gray = cv2.cvtColor(source_image_resized, cv2.COLOR_RGB2GRAY)
        # source_image_normalized = source_image_gray.astype(np.float32) / 255.0
        # source_image_normalized = np.expand_dims(source_image_normalized, axis=-1) # [H, W, 1]

        return dict(jpg=target_image_normalized, txt=prompt, hint=source_image_normalized)

if __name__ == '__main__':
    from PIL import Image

    try:
        dataset = MyControlNetDataset(json_file_path='test_dataset.json')
        print(f"dataset size: {len(dataset)}")

       
        if len(dataset) > 0:
            sample = dataset[0]
            print("First sample obtained:")
            print(f"  Text prompt (txt): {sample['txt']}")
            print(f"  Target image (jpg) shape: {sample['jpg'].shape}, dtype: {sample['jpg'].dtype}, min: {sample['jpg'].min()}, max: {sample['jpg'].max()}")
            print(f"  Control image (hint) shape: {sample['hint'].shape}, dtype: {sample['hint'].dtype}, min: {sample['hint'].min()}, max: {sample['hint'].max()}")
        else:
            print("No samples in the dataset.")
            
    except Exception as e:
        print(f"An error occurred while testing the dataset: {e}")
        import traceback
        traceback.print_exc()
