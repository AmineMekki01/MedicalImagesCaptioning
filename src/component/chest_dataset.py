from PIL import Image
from transformers import GPT2TokenizerFast
import numpy as np
import torch
from src.logs import logger

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


class Dataset:
    def __init__(self, dataFrame, transformations):
        self.dataFrame = dataFrame
        self.transformations = transformations

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        sample = self.dataFrame.iloc[idx, :]
        image = sample['image_path']
        caption = sample['image_caption']
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image_augmentation = self.transformations(image=image)
        image = image_augmentation['image']
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image, input_ids, labels


def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    image = torch.stack(image, dim=0)
    input_ids = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    labels = tokenizer.pad(
        {'input_ids': labels},
        padding='longest',
        return_attention_mask=False,
        return_tensors='pt'
    )['input_ids']
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask == 0] = -100
    return image, input_ids, labels
