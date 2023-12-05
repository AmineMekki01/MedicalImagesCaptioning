import gc
import numpy as np
import pandas as pd
from types import SimpleNamespace


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from src.component.model import VisionGPT2Model
from src.logs import logger


class Trainer:
    def __init__(self, model_config, train_config, model_output_path, dls, use_pretrained=False, fine_tune=False):
        self.model_output_path = model_output_path
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device
        logger.info(f"cuda available: {torch.cuda.is_available()}")

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

        self.sched = get_linear_schedule_with_warmup(self.optim,num_warmup_steps=0,num_training_steps=total_steps)

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity',
                      'val_loss', 'val_perplexity']] = None

        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[
                        0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])

    def save_checkpoint(self, model_path_output=None, epoch=None, best_pxp=None, best_epoch=None):
        if model_path_output is None:
            model_path_output = self.model_output_path
        checkpoint = {
            'epoch': epoch,
            'best_pxp': best_pxp,
            'best_epoch': best_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.sched.state_dict(),
        }
        torch.save(checkpoint, model_path_output)


    def load_checkpoint(self, model_path_output=None):
        if model_path_output is None:
            model_path_output = self.model_output_path
        checkpoint = torch.load(model_path_output)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.sched.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_pxp'], checkpoint['best_epoch']



    def train_one_epoch(self, epoch):

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
    def valid_one_epoch(self, epoch):

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

    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()

    def fit(self, resume):

        start_epoch, best_pxp, best_epoch = 0, 1e9, -1
        prog = tqdm(range(self.train_config.epochs))
        
        if resume:
            start_epoch, best_pxp, best_epoch = self.load_checkpoint()

        for epoch in range(start_epoch, self.train_config.epochs):
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

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_checkpoint(self.model_output_path, epoch=epoch, best_pxp=best_pxp, best_epoch=best_epoch)

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }

    @torch.no_grad()
    def generate_caption(self, image, max_tokens=50, temperature=1.0, deterministic=False):

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
