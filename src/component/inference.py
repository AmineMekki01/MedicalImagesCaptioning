
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import GPT2TokenizerFast
from src.component.model import VisionGPT2Model
from src.logs import logger


class Inference:
    def __init__(self, model_config, inference_config):
        self.model_config = model_config
        self.inference_config = inference_config
        self.model = None
        self.tokenizer = None
        self.device = self.inference_config.device
        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])
        self.load_tokenizer()
        self.load_model()

    def load_tokenizer(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self):
        self.model = VisionGPT2Model(self.model_config)
        self.model.load_state_dict(torch.load(
            self.inference_config.trained_model_path, map_location=torch.device(self.device)))
        self.model.to(device=self.device)
        self.model.eval()
        logger.info(
            f"Model in {self.inference_config.trained_model_path} loaded successfully")

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):

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
