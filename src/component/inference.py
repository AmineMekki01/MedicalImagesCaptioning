"""
Script : 
    inference.py

Description:
    The 'inference.py' module contains the Inference class designed for generating captions for new images using a trained VisionGPT2Model. This class is responsible for loading the trained model, setting up the necessary transformations for the input images, and providing methods to generate captions for single or multiple images.

Class:
    Inference: Manages the inference process for image captioning using a trained VisionGPT2Model.

Methods:
    __init__(self, model_config, inference_config):
        Initializes the Inference class with model and inference configurations. It loads the tokenizer and the trained model, and sets up image transformations.

    load_tokenizer(self):
        Loads the GPT2TokenizerFast tokenizer from HuggingFace and sets the pad token to the end-of-sequence token.

    load_model(self):
        Loads the trained VisionGPT2Model from the specified path in the inference configuration and sets it to evaluation mode.

    generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):
        Generates a caption for a single image. It takes the image path, processes the image, and feeds it to the model to generate a caption. The method offers customization options such as maximum tokens, temperature for sampling, and deterministic sampling.

    generate_caption_batch(self, image_paths, max_tokens=50, temperature=1.0, deterministic=False):
        Generates captions for a batch of images. This method is similar to `generate_caption` but operates on a list of image paths, making it more efficient for processing multiple images at once.

Dependencies:
    - torch, numpy: For tensor operations and numerical processing.
    - PIL: For image loading and conversion.
    - albumentations: For performing image transformations.
    - transformers: For utilizing GPT2TokenizerFast.
    - src.component.model: Contains the VisionGPT2Model.
    - src.logs: Logging module for tracking inference progress.
"""
from typing import Any, List, Optional

import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import GPT2TokenizerFast

from src.component.models.vit_gpt2 import VisionGPT2Model
from src.component.models.resnet_gpt2 import ResnetGPT2Model
from src.logs import logger

class Inference:
    def __init__(self, model_config: Any, inference_config: Any) -> None:
        self.model_config = model_config
        self.inference_config = inference_config
        self.model: Optional[VisionGPT2Model] = None
        self.tokenizer: Optional[GPT2TokenizerFast] = None
        self.device = self.inference_config.device
        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self) -> None:
        """ Load the GPT2TokenizerFast tokenizer from HuggingFace and set the pad token to the end-of-sequence token. """
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self) -> None:
        """ Load the trained VisionGPT2Model from the specified path in the inference configuration and set it to evaluation mode."""
        if self.model_config.encoder_type == 'resnet50':
            self.model = ResnetGPT2Model(self.model_config)
        elif self.model_config.encoder_type == 'gpt2':    
            self.model = VisionGPT2Model(self.model_config)
        model_path = str(self.inference_config.trained_model_path)
        model_path = model_path.replace('.pth', f'_{self.model_config.encoder_type}.pth')
        logger.info(f'Loading model from {model_path}')
        
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device(self.device)))
        self.model.to(device=self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_caption(self, image: str, max_tokens: int = 50, temperature: float = 1.0, deterministic: bool = False) -> str:
        """ Generate a caption for a single image.
        
        Args:
            image (str): The path to the image.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for the sampling distribution.
            deterministic (bool): Whether to use deterministic sampling or not.
        
        Returns:
            str: The generated caption.
        """
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = self.gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(self.device)
        sequence = torch.ones(1, 1).to(
            device=self.device).long() * self.tokenizer.bos_token_id

        caption = self.model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = self.tokenizer.decode(
            caption.numpy(), skip_special_tokens=True)

        return caption

    @torch.no_grad()
    def generate_caption_batch(self, image_paths: List[str], max_tokens: int = 50, temperature: float = 1.0, deterministic: bool = False) -> List[str]:
        """ Generate captions for a batch of images.
        
        Args:
            image_paths (List[str]): List of image paths.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): The temperature for the sampling distribution.
            deterministic (bool): Whether to use deterministic sampling or not.
        
        Returns:
            List[str]: List of generated captions.
        """
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        batch_images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            transformed_image = self.gen_tfms(image=image)['image']
            batch_images.append(transformed_image)

        batch_images_tensor = torch.stack(batch_images).to(self.device)
        batch_size = len(batch_images_tensor)
        sequence = torch.ones(batch_size, 1).to(self.device).long() * self.tokenizer.bos_token_id

        captions = self.model.generate(
            image=batch_images_tensor,
            sequence=sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )

        decoded_captions = [self.tokenizer.decode(caption, skip_special_tokens=True) for caption in captions]
        return decoded_captions
    