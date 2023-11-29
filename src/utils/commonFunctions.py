"""
Script_name : common_functions.py
Author_name : Amine MEKKI

Description:
In this file i will put the common used functions so as i dont repeat my self.
"""

import os
import yaml
import pandas as pd
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path
from box import ConfigBox
from sklearn.model_selection import train_test_split

from src.logs import logger

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
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
            logger.info(f"Yaml file : {os.path.normpath(path_to_yaml)} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e  


@ensure_annotations
def create_directories(path_to_directories : list, verbose : bool = True):
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
def split_data(data : pd.DataFrame, validation_size : float = 0.1, test_size : float = 0.1, random_state : int = 42):
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
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    train, validation = train_test_split(train, test_size=validation_size, random_state=random_state)
    return train, validation, test  