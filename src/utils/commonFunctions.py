"""
Script_name : commonFunctions.py
Author_name : Amine MEKKI

Description:
In this file i will put the common used functions so as i dont repeat my self.
"""

from src.logs import logger
import os
import yaml
import pandas as pd
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path
from box import ConfigBox
from sklearn.model_selection import train_test_split
# from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    This function reads a yaml file and returns a ConfigBox object. 

    Parameters
    ----------
    path_to_yaml : Path
        path to yaml file.

    Raises:
        ValueError: if yaml file is empty.
        e: if any other error occurs.

    Returns:
    -------
        ConfigBox : ConfigBox object.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(
                f"Yaml file : {os.path.normpath(path_to_yaml)} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    This function creates directories if they dont exist.

    Parameters
    ----------
    path_to_directories : list
        list of paths to directories.   
    verbose : bool, optional
        if True, print the created directories, by default True 

    Returns
    -------
    None
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at : {os.path.normpath(path)}")


@ensure_annotations
def split_data(data: pd.DataFrame, validation_size: float = 0.1, test_size: float = 0.1, random_state: int = 42):
    """
    This function splits data into train and test sets.

    Parameters
    ----------
    data : list
        list of data to split.
    train_size : float, optional
        train set size, by default 0.8
    random_state : int, optional
        random state, by default 42

    Returns
    -------
    train, test : list, list
        train and test sets.
    """
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(
        train, test_size=validation_size, random_state=random_state)
    return train, validation, test


def save_splitted_data(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, processed_data_path: Path):
    """
    This function saves the splitted data into csv files.

    Parameters
    ----------
    train : pd.DataFrame
        train set.
    validation : pd.DataFrame
        validation set.
    test : pd.DataFrame
        test set.

    Returns
    -------
    None
    """
    logger.info(f"amine : {processed_data_path}")
    train_path = processed_data_path / "train.csv"
    test_path = processed_data_path / "test.csv"
    validation_path = processed_data_path / "validation.csv"
    train.to_csv(train_path, index=False, sep=';')
    validation.to_csv(validation_path, index=False, sep=';')
    test.to_csv(test_path, index=False, sep=';')
    logger.info("data splitted successfully.")


def evaluate_model_generator(reference_caption, generated_caption):
    """
    This function evaluates the model using the generated caption and the reference caption.

    Parameters
    ----------
    reference_caption : string
        reference caption.  
    generated_caption : string  
        generated caption.

    Returns 
    -------
    cider_score, rouge1, rouge2, rougeL, bleu : float, float, float, float, float
        cider score, rouge1 score, rouge2 score, rougeL score and bleu score.
    """
    try:
        rouge1, rouge2, rougeL = calculate_rouge_score(
            reference_caption, generated_caption)
    except Exception as e:
        logger.error(e)
        rouge1, rouge2, rougeL = 0, 0, 0
    try:
        bleu_score = calculate_blue_score(reference_caption, generated_caption)
    except Exception as e:
        logger.error(e)
        bleu_score = 0
        
    # multiply by 100 and round to 2 decimal places
    rouge1 = round(rouge1 * 100, 2)
    rouge2 = round(rouge2 * 100, 2)
    rougeL = round(rougeL * 100, 2)
    bleu_score = round(bleu_score * 100, 2) 
    return rouge1, rouge2, rougeL, bleu_score


def calculate_rouge_score(reference_caption: str, generated_caption: str):
    """ 
    Calculate the ROUGE score for a given generated caption and reference caption.  

    Parameters  
    ----------  
    reference_caption : str  
        Reference caption.  
    generated_caption : str 
        Generated caption.

    Returns 
    ------- 
    rouge1, rouge2, rougeL : float, float, float 
        ROUGE-1, ROUGE-2 and ROUGE-L scores. 
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(generated_caption, reference_caption)
    rouge1 = rouge_scores[0]['rouge-1']['f']
    rouge2 = rouge_scores[0]['rouge-2']['f']
    rougeL = rouge_scores[0]['rouge-l']['f']
    return rouge1, rouge2, rougeL


def calculate_blue_score(generated_caption: str, reference_caption: str):
    """
    Calculate the BLEU score for a given generated caption and reference caption.   

    Parameters  
    ----------  
    generated_caption : str  
        Generated caption.  
    reference_caption : str 
        Reference caption.

    Returns 
    ------- 
    float 
        BLEU score. 
    """
    generated_tokens = word_tokenize(generated_caption.lower())
    reference_tokens = word_tokenize(reference_caption.lower())
    chencherry = SmoothingFunction()

    blue_score = sentence_bleu(
        [reference_tokens], generated_tokens, smoothing_function=chencherry.method1)

    return blue_score


def calculate_bert_score():
    pass


def read_train_val_csv(train_csv_path: Path, val_csv_path: Path):
    """
    This function reads the train and validation csv files.

    Parameters
    ----------
    train_csv_path : Path
        path to the train csv file.
    val_csv_path : Path
        path to the validation csv file.

    Returns
    -------
    train, validation : pd.DataFrame, pd.DataFrame
        train and validation dataframes.
    """
    train = pd.read_csv(train_csv_path, sep=';')
    validation = pd.read_csv(val_csv_path, sep=';')
    return train, validation


def create_image_path_column(images_directory_path: Path):
    """
    list all the paths in a directory and then create a dataframe with the image path column.

    Parameters  
    ----------  
    images_directory_path : Path    
        The path to the images directory.   
    
    Returns 
    ------- 
    pd.DataFrame
        The dataframe containing the image paths.   
    """
    image_paths = list(images_directory_path.glob('*.jpg'))
    image_paths = [str(path) for path in image_paths]
    image_paths = pd.DataFrame(image_paths, columns=['image_path'])
    return image_paths


def filter_rows_of_missing_images(data: pd.DataFrame, images_directory_path: Path):
    """
    This function filters the rows of missing images.\
        
    Parameters  
    ----------  
    data : pd.DataFrame  
        The dataframe to filter.
    images_directory_path : Path    
        The path to the images directory.   
    
    Returns
    ------- 
    pd.DataFrame
        The filtered dataframe. 
    """
    image_paths = list(images_directory_path.glob('*.jpg'))
    image_paths = [str(path) for path in image_paths]
    data = data[data['image_path'].isin(image_paths)]
    return data
