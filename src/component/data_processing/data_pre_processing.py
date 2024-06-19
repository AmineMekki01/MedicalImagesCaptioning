"""
Script : 
    dataPreProcessing.py
    
Description :
    The 'dataPreProcessing.py' module offers a suite of functions dedicated to preprocessing data for dataset creation in machine learning models. These functions are instrumental in transforming raw data into a structured and cleaned format, suitable for training and evaluation purposes.

Functions:
    dataUploadMerge(projections_path: Path, reports_path: Path) -> pd.DataFrame:
        Merges projection and report data into a single DataFrame. It requires paths to the projections and reports data as inputs.

    create_image_caption_dict(merged_data: pd.DataFrame) -> Dict:
        Converts a DataFrame into a dictionary mapping image filenames to their captions. It processes the 'findings' and 'impression' fields to extract meaningful captions.

    cleanse_data(data: Dict, images_base_path: Path) -> Dict:
        Cleanses the data by converting text to lowercase and filtering out short words. It returns a dictionary with processed captions and updates image paths to include the base path.

    decontracted(phrase: str) -> str:
        Expands contracted words in a given phrase (e.g., converting "won't" to "will not"). This function is used in text preprocessing to standardize the textual data.

    preprocess_text(data: pd.DataFrame) -> list:
        Performs a series of text preprocessing steps on a DataFrame's columns, including pattern removal, special character cleaning, and text normalization.

    preprocessData(dataframe: pd.DataFrame, images_base_path: Path, processed_data_path: Path) -> pd.DataFrame:
        Orchestrates the overall preprocessing of the dataset. It combines creating captions, cleansing data, and text preprocessing into a streamlined process.

    convert_txt_to_csv(text_folder_path: Path, image_path_prefix: Path, output_path: Path) -> pd.DataFrame:
        Converts text files into a CSV format, facilitating easier data handling. It processes text files containing image paths and captions, and outputs a DataFrame with the structured data.

Dependencies:
    - pandas, numpy: For data manipulation and numerical computations.
    - re: For regular expression operations.
    - pathlib: For handling file and directory paths.
    - tqdm: For providing progress bars during data processing.
    - src.logs: For logging information during the preprocessing steps.
    - src.utils.commonFunctions: Provides utility functions like data splitting and saving.
"""


import pandas as pd
import numpy as np
import re
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

from src.logs import logger
from src.utils.common_functions import split_data, save_splitted_data, filter_rows_of_missing_images


def dataUploadMerge(projections_path: Path, reports_path: Path) -> pd.DataFrame:
    """ Merges the projections and reports data into a single DataFrame.

    Args:   
        projections_path (Path): The path to the projections data.   
        reports_path (Path): The path to the reports data.

    Returns:
        merged_data (pd.DataFrame): A DataFrame containing the merged data.
    """
    projections_data = pd.read_csv(projections_path)
    reports_data = pd.read_csv(reports_path)
    merged_data = pd.merge(projections_data, reports_data, on='uid')
    return merged_data


def create_image_caption_dict(merged_data: pd.DataFrame) -> Dict[str, List[str]]:
    """ Creates a dictionary of image filenames and their corresponding captions.

    Args:
        merged_data (pd.DataFrame): A DataFrame containing the columns 'filename', 'findings', and 'impression'.

    Returns:
        data (Dict): A dictionary where keys are filenames and values are lists of captions.
    """
    data: Dict[str, List[str]] = {}

    for i in range(len(merged_data)):
        filename = merged_data.loc[i, 'filename']
        captions = merged_data.loc[i, 'findings']

        if filename not in data:
            data[filename] = []

        # Check if 'findings' is null
        if isinstance(captions, float) and np.isnan(captions):
            captions = merged_data.loc[i, 'impression']

        # Process captions that start with a number followed by a period
        if isinstance(captions, str) and re.match(r'^\d+\.', captions):
            data[filename].append(captions.split('. ')[1])
        else:
            if data[filename]:
                data[filename][-1] += " " + captions
            else:
                data[filename].append(captions)
    return data


def cleanse_data(data: Dict[str, List[str]], images_base_path: Path) -> pd.DataFrame:
    """ Cleanses the data by converting strings to lowercase and removing short words.

    Args:
        data (Dict): A dictionary where keys are filenames and values are lists of captions.

    Returns:
        dataframe (pd.DataFrame): A DataFrame containing the cleansed data.
    """
    # cleansed_data: Dict[str, List[str]] = {}
    cleansed_data = {}
    for key, values in data.items():
        for value in values:
            cleansed_line = ""
            if isinstance(value, str):
                for word in value.split():
                    if len(word) >= 2:
                        cleansed_line += word.lower() + " "
                if key not in cleansed_data:
                    cleansed_data[key] = []
                cleansed_data[key].append(cleansed_line.strip())

    dataframe = pd.DataFrame(cleansed_data.items(), columns=[
                             'image_path', 'image_caption'])

    if isinstance(dataframe['image_caption'][0], list):
        dataframe['image_caption'] = dataframe['image_caption'].apply(
            lambda x: x[0])
        dataframe['image_path'] = dataframe['image_path'].apply(
            lambda x: images_base_path / x)
    else:
        print("no")

    return dataframe


def decontracted(phrase: str) -> str:
    """ Performs text decontraction of words like won't to will not.

    Args: 
        phrase (str): A string to be processed.   

    Returns:    
        phrase (str): The processed string.
    """
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(data: pd.DataFrame) -> List[str]:
    """ Extracts the information from the data and does text preprocessing on them.

    Args: 
        data (pd.DataFrame): A DataFrame containing the columns 'filename', 'findings', and 'impression'.    

    Returns:
        preprocessed (List[str]): A list of preprocessed sentences.   

    """
    preprocessed: List[str] = []

    for sentence in tqdm(data.values):
        # Removing patterns like "1."
        sentence = re.sub(r"\b\d\.", "", sentence)

        # Removing words like "xxxx" (case-insensitive)
        sentence = re.sub(r"xxxx", "", sentence)

        # Removing all special characters except for full stop
        sentence = re.sub(r"[^.a-zA-Z]", " ", sentence)

        # Replacing specific terms
        replacements = {
            '&': 'and',
            '@': 'at',
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine',
            'year old': '',
            'yearold': ''
        }
        for old, new in replacements.items():
            sentence = sentence.replace(old, new)

        # Decontraction and other text processing
        sentence = decontracted(sentence)
        sentence = sentence.strip().lower()
        sentence = " ".join(sentence.split())

        if sentence == "":
            sentence = np.nan
        preprocessed.append(sentence)
    return preprocessed


def preprocessData(dataframe: pd.DataFrame, images_base_path: Path, processed_data_path: Path) -> pd.DataFrame:
    """ Preprocess the data by extracting the information from the data and doing text preprocessing on them.

    Args: 
        dataframe (pd.DataFrame): A DataFrame containing the columns 'filename', 'findings', and 'impression'.    
        images_base_path (Path): The path to the images.    

    Returns:    
        clean_dataframe (pd.DataFrame): A DataFrame containing the preprocessed data.
    """
    dataDict = create_image_caption_dict(dataframe)
    clean_dataframe = cleanse_data(dataDict, images_base_path)
    print(clean_dataframe.columns)
    clean_dataframe["image_caption"] = preprocess_text(
        clean_dataframe["image_caption"])
    clean_dataframe.dropna(inplace=True)
    data_contains_xxxx2 = clean_dataframe["image_caption"].str.contains(
        'xxxx', case=False, regex=False).any()
    logger.info(f"Data contains xxxx: {data_contains_xxxx2}")
    train, validation, test = split_data(clean_dataframe)
    save_splitted_data(train, validation, test, processed_data_path)
    return clean_dataframe


def convert_txt_to_csv(text_folder_path: Path, image_path_prefix: Path, output_path: Path) -> pd.DataFrame:
    """ Converts a text file to a csv file.

    Args: 
        text_file_path (Path): The path to the text file.    
        image_path_prefix (Path): The path to the images.    
        output_path (Path): The path to the output csv file.    

    Returns:    
        df_filtered (pd.DataFrame): A DataFrame containing the filtered data.    
    """
    # get all the text files
    text_files = list(text_folder_path.glob('*.txt'))
    for text_file_path in text_files:
        with open(text_file_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        final_lines = []
        for line in lines:

            line = line.strip().split('\t', 1)
            if len(line) != 2:
                continue
            final_lines.append(line)

        test_final_lines = []
        for line in final_lines:

            test_final_lines.append(
                [str(image_path_prefix / (line[0] + '.jpg')), line[1]]
            )

        df = pd.DataFrame(test_final_lines, columns=[
                          'image_path', 'image_caption'])
        df_filtered = filter_rows_of_missing_images(df, image_path_prefix)

        if "train" in str(text_file_path):
            final_path = output_path / "train.csv"
        elif "validation" in str(text_file_path):
            final_path = output_path / "validation.csv"
        elif "test" in str(text_file_path):
            final_path = output_path / "test.csv"
        else:
            logger.exception("The text file path is not valid.")
        logger.info(f"Saving csv to {final_path}")
        df_filtered.to_csv(final_path, index=False, sep=';')

    return df_filtered
