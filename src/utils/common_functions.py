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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """ Read yaml file and return a ConfigBox object.
    
    Args:
        path_to_yaml (Path): Path to the yaml file.
        
    Returns:
        ConfigBox: A ConfigBox object containing the yaml file content.
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
    """ Create Directories. """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at : {os.path.normpath(path)}")


@ensure_annotations
def split_data(data: pd.DataFrame, validation_size: float = 0.1, test_size: float = 0.1, random_state: int = 42) -> tuple:
    """ Split the data into train, validation and test sets.
    
    Args:
        data (pd.DataFrame): The data to split.
        validation_size (float): The size of the validation set.
        test_size (float): The size of the test set.
        random_state (int): The random state for reproducibility.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test sets.
    """
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(
        train, test_size=validation_size, random_state=random_state)
    return train, validation, test


def save_splitted_data(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, processed_data_path: Path) -> None:
    """ Save the splitted data into csv files.
    
    Args:
        train (pd.DataFrame): The training set.
        validation (pd.DataFrame): The validation set.
        test (pd.DataFrame): The test set.
        processed_data_path (Path): The path to save the splitted data.
    
    Returns:
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


def evaluate_model_generator(reference_caption, generated_caption) -> tuple:
    """ Evaluate the generated caption using Rouge and BLEU scores.
    
    Args:
        reference_caption (str): The reference caption.
        generated_caption (str): The generated caption.
    
    Returns:
        tuple: The Rouge1, Rouge2, RougeL, BLEU scores.
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
    
    try:
        cider_score = compute_cider_gpt2(reference_caption, generated_caption)
        print(cider_score)
    except Exception as e:  
        logger.error(e)
        cider_score = 0 
        
    rouge1 = round(rouge1 * 100, 2)
    rouge2 = round(rouge2 * 100, 2)
    rougeL = round(rougeL * 100, 2)
    bleu_score = round(bleu_score * 100, 2) 
    return rouge1, rouge2, rougeL, bleu_score, cider_score


def calculate_rouge_score(reference_caption: str, generated_caption: str):
    """ Calculate the Rouge scores for the generated caption.
    
    Args:
        reference_caption (str): The reference caption.
        generated_caption (str): The generated caption.
    
    Returns:
        tuple: The Rouge1, Rouge2, RougeL scores.
    """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(hyps=generated_caption, refs=reference_caption)
    rouge1 = rouge_scores[0]['rouge-1']['f']
    rouge2 = rouge_scores[0]['rouge-2']['f']
    rougeL = rouge_scores[0]['rouge-l']['f']
    return rouge1, rouge2, rougeL


def calculate_blue_score(reference_caption: str,  generated_caption: str) -> float:
    """ Calculate the BLEU score for the generated caption.
    
    Args:
        reference_caption (str): The reference caption.
        generated_caption (str): The generated caption.
    
    Returns:
        float: The BLEU score.
    """
    generated_tokens = word_tokenize(generated_caption.lower())
    reference_tokens = word_tokenize(reference_caption.lower())
    chencherry = SmoothingFunction()

    blue_score = sentence_bleu(
        references=[reference_tokens], hypothesis=         generated_tokens, smoothing_function=chencherry.method1)

    return blue_score


def gpt2_tokenize(captions) -> list:
    """ Tokenize the captions using GPT-2 tokenizer.

    Args:
        captions (List[str]): The captions to tokenize.
    
    Returns:
        List[str]: The tokenized captions.
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return [" ".join(tokenizer.tokenize(caption)) for caption in captions]

def compute_cider_gpt2(reference_captions : str, generated_caption : str) -> float:
    """ Compute the CIDEr score for the generated caption.
    
    Args:
        reference_captions (List[str]): The reference captions.
        generated_caption (str): The generated caption.
    
    Returns:
        float: The CIDEr score.
    """
    reference_captions = [reference_captions]
    generated_caption = [generated_caption]
    
    vectorizer = TfidfVectorizer()

    tokenized_candidates = gpt2_tokenize(generated_caption)
    tokenized_references = gpt2_tokenize(reference_captions)

    combined_captions = tokenized_candidates + tokenized_references
    tfidf_matrix = vectorizer.fit_transform(combined_captions)
    cos_similarities = cosine_similarity(tfidf_matrix[:len(tokenized_candidates)], tfidf_matrix[len(tokenized_candidates):])
    cider_scores = cos_similarities.mean(axis=1)
    return cider_scores



def read_train_val_csv(train_csv_path: Path, val_csv_path: Path) -> tuple:
    """ Read the train and validation csv files.
    
    Args:
        train_csv_path (Path): The path to the train csv file.
        val_csv_path (Path): The path to the validation csv file.
    
    Returns:
        tuple: The train and validation dataframes.
    """
    train = pd.read_csv(train_csv_path, sep=';')
    validation = pd.read_csv(val_csv_path, sep=';')
    return train, validation


def create_image_path_column(images_directory_path: Path) -> pd.DataFrame:
    """ Create a dataframe containing the image paths.
    
    Args:
        images_directory_path (Path): The path to the images directory.
    
    Returns:
        image_paths_df (pd.DataFrame): The dataframe containing the image paths.
    """
    image_paths = list(images_directory_path.glob('*.jpg'))
    image_paths = [str(path) for path in image_paths]
    image_paths_df = pd.DataFrame(image_paths, columns=['image_path'])
    return image_paths_df


def filter_rows_of_missing_images(data: pd.DataFrame, images_directory_path: Path) -> pd.DataFrame:
    """ Filter the rows of the dataframe that contain missing images.
    
    Args:
        data (pd.DataFrame): The dataframe to filter.
        images_directory_path (Path): The path to the images directory.
    
    Returns:
        data (pd.DataFrame): The filtered dataframe.
    """
    image_paths = list(images_directory_path.glob('*.jpg'))
    image_paths = [str(path) for path in image_paths]
    data = data[data['image_path'].isin(image_paths)]
    return data
