"""
Script_name : common_functions.py
Author_name : Amine MEKKI

Description:
In this file i will put the common used functions so as i dont repeat my self.
"""

import os
import yaml
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from pathlib import Path
from box import ConfigBox

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
            