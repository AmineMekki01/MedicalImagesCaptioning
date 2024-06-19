"""
Script : 
    imageTransform.py

Description : 
    The 'imageTransform.py' module provides a set of image transformations using the Albumentations library, which are crucial for preparing images for the dataset creation and training process in a machine learning pipeline. These transformations are designed to augment and normalize the images, making the model more robust to variations in the input data.

Transformations:
    sample_tfms: A list of basic transformations applied to each image sample. It includes operations like horizontal flipping, random brightness and contrast adjustments, color jitter, shift-scale-rotate transformations, and hue-saturation-value adjustments. These transformations enhance the diversity of the dataset by introducing variations in the images.

    train_tfms: Composed transformations for the training dataset. It includes all transformations defined in `sample_tfms` and additional steps like resizing the image to 224x224 pixels and normalizing pixel values. The normalization uses a mean and standard deviation of [0.5, 0.5, 0.5] and converts the image to a PyTorch tensor using `ToTensorV2`.

    valid_tfms: Composed transformations for the validation dataset. It omits the augmentations present in `sample_tfms` and only includes resizing and normalization to maintain the originality of validation images. This is crucial for evaluating the model's performance on unaltered images.

Dependencies:
    - albumentations: A fast image augmentation library that is compatible with PyTorch and provides a wide range of transformation techniques.
    - albumentations.pytorch: Provides the `ToTensorV2` transformation for converting images to PyTorch tensors.
"""


import albumentations as A
from albumentations.pytorch import ToTensorV2



sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])