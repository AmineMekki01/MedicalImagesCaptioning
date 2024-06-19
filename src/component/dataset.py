"""
Script : 
    dataset.py
    
Description :
    The 'dataset.py' module contains the Dataset class and a custom collate function designed for loading and preprocessing data in a machine learning pipeline, particularly for tasks involving image captioning. The Dataset class is responsible for handling the data frame containing image paths and captions, applying transformations to images, and preparing text data for training. The collate function is essential for batching data samples effectively during training.

Classes:
    Dataset: A custom dataset class for handling image-caption pairs.
        - __init__(self, dataFrame, transformations): Initializes the Dataset with a data frame containing image paths and captions and the specified image transformations.
        - __len__(self): Returns the length of the dataset.
        - __getitem__(self, idx): Retrieves an image-caption pair from the dataset at the specified index, applies transformations to the image, and prepares the text data.

Functions:
    collate_fn(batch): A custom collate function for the dataloader.
        - Parameters: batch (list) - A list of samples from the dataset.
        - Returns: A tuple containing tensors for images, input_ids, and labels. It handles padding for text data and prepares label tensors for the model.

Dependencies:
    - PIL: For loading and processing images.
    - transformers: Provides the GPT2TokenizerFast for tokenizing captions.
    - numpy, torch: For numerical operations and tensor manipulations.
    - src.logs: For logging information during dataset processing.

"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import GPT2TokenizerFast
from typing import Tuple, List
from albumentations.core.composition import Compose

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


class Dataset:
    def __init__(self, dataFrame : pd.DataFrame, transformations : Compose):
        self.dataFrame = dataFrame
        self.transformations = transformations

    def __len__(self) -> int:
        """ Returns the length of the dataset. """
        return len(self.dataFrame)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], List[int]]:
        """ Retrieves an image-caption pair from the dataset at the specified index.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            image (torch.Tensor): The image tensor.
            input_ids (List[int]): The input_ids tensor.
            labels (List[int]): The labels tensor.
        """
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


def collate_fn(batch: List[Tuple[torch.Tensor, List[int], List[int]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ A custom collate function for the dataloader.
    
    Args:
        batch (List[Tuple[torch.Tensor, List[int], List[int]]]): A list of samples from the dataset.

    Returns:
        image (torch.Tensor): The image tensor.
        input_ids (torch.Tensor): The input_ids tensor.
        labels (torch.Tensor): The labels tensor.
    """
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
