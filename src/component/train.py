"""
Script : 
    train.py
    
Description :
    The 'train.py' module contains the Trainer class, which is responsible for the training loop of a Vision-GPT2 model. The Trainer is initialized with model and training configurations, data loaders, and options for using pretrained models and fine-tuning. This module facilitates the entire training process, including model training, validation, and caption generation for images.

Class:
    Trainer: Manages the training and validation of the Vision-GPT2 model.

Methods:
    __init__(self, model_config, train_config, model_output_path, dls, use_pretrained=False, fine_tune=False):
        Initializes the Trainer class with model configuration, training configuration, model output path, data loaders, and options for using a pretrained model and fine-tuning.

    save_model(self, model_path_output=None):
        Saves the trained model to a specified path.

    load_best_model(self, model_path_output=None):
        Loads the best performing model from a specified path.

    train_one_epoch(self, epoch: int):
        Conducts the training of the model for one epoch and returns the training loss.

    valid_one_epoch(self, epoch: int):
        Conducts the validation of the model for one epoch and returns the validation perplexity.

    clean(self):
        Clears the GPU memory to optimize memory usage.

    fit(self):
        Executes the complete training process for the specified number of epochs and manages the freezing and unfreezing of model layers based on the training configuration.

    generate_caption(self, image: str, max_tokens: int = 50, temperature: float = 1.0, deterministic: bool = False):
        Generates a caption for a given image using the trained model. The function allows customization of the maximum tokens, temperature for generation, and whether to use deterministic sampling.

Dependencies:
    - torch, numpy, pandas: For numerical and tensor operations.
    - transformers: For utilizing GPT2TokenizerFast.
    - albumentations: For image transformations.
    - PIL: For image processing.
    - src.component.model: Contains the VisionGPT2Model class.
    - src.logs: Logging module for tracking progress and activities.

"""

import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from src.component.vitGPT2Model import VisionGPT2Model
from src.component.resnetGPT2Model import ResnetGPT2Model   
from src.logs import logger
from typing import Tuple, Dict, Optional, Any


class Trainer:
    def __init__(self, model_config: Any, train_config: Any, model_output_path: str, dls: Tuple[Any, Any], use_pretrained: bool = False, fine_tune: bool = False) -> None:
        self.model_output_path = model_output_path
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device
        logger.info(f"cuda available: {torch.cuda.is_available()}")
        
        if self.model_config.encoder_type == 'resnet50':
            if use_pretrained:
                self.model = ResnetGPT2Model.from_pretrained(config = model_config).to(self.device)
            else:
                self.model = ResnetGPT2Model(config = model_config).to(self.device)
  
        elif self.model_config.encoder_type == 'gpt2':  
            if use_pretrained:
                self.model = VisionGPT2Model.from_pretrained(
                    model_config).to(self.device)
            else:
                self.model = VisionGPT2Model(model_config).to(
                    self.device)

        self.model.pretrained_layers_trainable(trainable=not fine_tune)

        logger.info(
            f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scaler = GradScaler()

        self.train_dl, self.val_dl = dls

        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.learning_rate,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

        # self.sched = get_linear_schedule_with_warmup(self.optim,num_warmup_steps=0,num_training_steps=total_steps)

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity',
                      'val_loss', 'val_perplexity']] = None

        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

    def save_model(self, model_path_output: Optional[str] = None) -> None:
        """
        Saves the model to the specified path.  
        """
        if model_path_output is None:
            model_path_output = self.model_output_path
        
        model_path_output = model_path_output.replace('.pth', f'_{self.model_config.encoder_type}.pth')
        sd = self.model.state_dict()
        torch.save(sd, model_path_output)

    def load_best_model(self, model_path_output: Optional[str] = None) -> None:
        """
        Loads the best model from the specified path.   
        """
        if model_path_output is None:
            model_path_output = self.model_output_path
        
        model_path_output = model_path_output.replace('.pth', f'_{self.model_config.encoder_type}.pth')
        sd = torch.load(model_path_output)
        self.model.load_state_dict(sd)

    def train_one_epoch(self, epoch: int) -> float:
        """
        Trains the model for one epoch. 
    
        Parameters:
        ----------- 
        epoch (int): 
            The current epoch.  
        
        Returns:    
        --------
        float: 
            The training loss.
        """
        prog = tqdm(self.train_dl, total=len(self.train_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()

                prog.set_description(f'train loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        self.metrics.loc[epoch, ['train_loss', 'train_perplexity']] = (
            train_loss, train_pxp)

    @torch.no_grad()
    def valid_one_epoch(self, epoch: int) -> float:
        """
        Validates the model for one epoch.
    
        Parameters: 
        ----------- 
        epoch (int): 
            The current epoch.  
        
        Returns:    
        --------    
        float: 
            The validation perplexity.
        """

        prog = tqdm(self.val_dl, total=len(self.val_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image, input_ids, labels)
                running_loss += loss.item()

                prog.set_description(f'valid loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        self.metrics.loc[epoch, ['val_loss', 'val_perplexity']] = (
            val_loss, val_pxp)

        return val_pxp

    def clean(self) -> None:
        """
        Cleans the GPU memory. Only 16Gb LoL.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self) -> Dict[str, float]:
        """
        Trains the model.
        """
        best_pxp = 1e9
        best_epoch = -1
        patience_counter = 0 
        patience = 3  
        min_delta = 0.01  
        
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:
            print("doing epoch", epoch)
            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            prog.set_description('validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            if pxp < best_pxp - min_delta:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model(self.model_output_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Stopping early at epoch {epoch}. Best epoch: {best_epoch} with perplexity: {best_pxp}.")
                break

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image: str, max_tokens: int = 50, temperature: float = 1.0, deterministic: bool = False) -> str:
        """
        Generate a caption for a single image.  
    
        Parameters: 
        ----------- 
        image (str): 
            The path to the image.  
        
        max_tokens (int):   
            The maximum number of tokens to generate.   
        
        temperature (float):    
            The temperature to use for the generation.  
        
        deterministic (bool):   
            Whether to use deterministic sampling or not.   
        
        Returns:
        --------    
        str: 
            The generated caption decoded.  
        """
        self.model.eval()

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
