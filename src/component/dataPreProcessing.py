import pandas as pd
import numpy as np
import re
from typing import Dict
from pathlib import Path
from tqdm import tqdm   


def dataUploadMerge(projections_path : Path, reports_path : Path) -> pd.DataFrame:
    """
    Merges the projections and reports data into a single DataFrame.

    Parameters: 
    ----------  
    projections_path (Path):    
        The path to the projections data.   
    reports_path (Path):    
        The path to the reports data.
    
    Returns:    
    ------- 
    pd.DataFrame:   
        A DataFrame containing the merged data.
    """
    projections_data = pd.read_csv(projections_path)
    reports_data = pd.read_csv(reports_path)
    merged_data = pd.merge(projections_data, reports_data, on='uid')
    return merged_data


def create_image_caption_dict(merged_data : pd.DataFrame) -> Dict:
    """
    Creates a dictionary of image filenames and their corresponding captions.

    Parameters:
    ----------
    merged_data (pd.DataFrame):
        A DataFrame containing the columns 'filename', 'findings', and 'impression'.

    Returns:
    -------
        Dict: 
            A dictionary where keys are filenames and values are lists of captions.
        int: A count of entries with problematic captions.
    """
    data = {}

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

def cleanse_data(data : Dict, images_base_path : Path) -> Dict:
    """
    Cleanses the data by converting strings to lowercase and removing short words.

    Parameters:
    ----------
    data (Dict): 
        A dictionary where keys are filenames and values are lists of captions.

    Returns:
    -------
    Dict: 
        A cleansed dictionary where each string is processed to remove short words and convert to lowercase.
    """
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
                
    dataframe = pd.DataFrame(cleansed_data.items(), columns=['image_path', 'image_caption'])
    
    if isinstance(dataframe['image_caption'][0], list):
        dataframe['image_caption'] = dataframe['image_caption'].apply(lambda x: x[0])
        dataframe['image_path'] = dataframe['image_path'].apply(lambda x: images_base_path / x)
    else: 
        print("no")
        
    return dataframe

def decontracted(phrase : str) -> str:
    """
    Performs text decontraction of words like won't to will not.
    
    Parameters: 
    ----------
    phrase (str): 
        A string to be processed.   
        
    Returns:    
    -------
    str: 
        A string with decontraction performed.
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

def preprocess_text(data : pd.DataFrame):
    """
    Extracts the information from the data and does text preprocessing on them.
    
    Parameters: 
    ----------
    data (pd.DataFrame): 
        A DataFrame containing the columns 'filename', 'findings', and 'impression'.    
    
    Returns:    
    -------
    list: 
        A list of preprocessed sentences.   
    
    """
    preprocessed = []

    for sentence in tqdm(data.values):
        # Removing patterns like "1."
        sentence = re.sub(r"\b\d\.", "", sentence)

        # Removing words like "xxxx" (case-insensitive)
        sentence = re.sub(r"(?i)\bxxxx\b", "", sentence)

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


def preprocessData(dataframe : pd.DataFrame , images_base_path : Path) -> pd.DataFrame:
    """
    Preprocesses the data by extracting the information from the data and doing text preprocessing on them.
    
    Parameters: 
    ----------
    dataframe (pd.DataFrame): 
        A DataFrame containing the columns 'filename', 'findings', and 'impression'.    
    images_base_path (Path): 
        The path to the images.    
    
    Returns:    
    -------
    pd.DataFrame: 
        A DataFrame containing the preprocessed data.   
    
    """
    dataDict = create_image_caption_dict(dataframe)
    dataframeCleanse = cleanse_data(dataDict, images_base_path)
    dataframeCleanse['image_caption'] = preprocess_text(dataframeCleanse['image_caption'])
    dataframeCleanse.dropna(inplace=True)
    return dataframeCleanse