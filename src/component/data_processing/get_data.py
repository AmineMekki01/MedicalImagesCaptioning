"""
Script : 
    getData.py
Description :
    This script is used to get the data and best model model from google drive and prepare it for the training process.

"""

import os
import requests
from tqdm import tqdm
import patoolib



def download_file_from_google_drive(id : str, destination : str) -> None:
    """ Download file from google drive.
    
    Args:
        id (str) : file id
        destination (str) : destination path
        
    Returns:
        None
    
    """
    URL = "https://docs.google.com/uc?export=download&confirm=1"

    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)    

def get_confirm_token(response : requests.Response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response : requests.Response, destination : str) -> None:
    """ Save response content.
    
    Args:
        response (requests.Response) : response
        destination (str) : destination path
    
    Returns:
        None
    """
    CHUNK_SIZE = 32768
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(CHUNK_SIZE):
            size = f.write(chunk)
            bar.update(size)

def extract_rar(file_path : str, dest_path : str) -> None:
    """ Extract rar file.
    
    Args:
        file_path (str) : file path
        dest_path (str) : destination path
        
    Returns:
        None
    """
    try:
        patoolib.extract_archive(file_path, outdir=dest_path)
    except patoolib.util.PatoolError as e:
        print(f"Extraction failed: {e}")

def download_and_extract(file_id : str, dest_folder : str) -> None:
    """ Download and extract file.
    
    Args:
        file_id (str) : file id
        dest_folder (str) : destination folder
    
    Returns:
        None
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        
    rar_path = os.path.join(dest_folder, 'raw.rar')
    download_file_from_google_drive(file_id, rar_path)
    
    if not os.path.exists(rar_path):
          (file_id, rar_path)
    extract_rar(rar_path, dest_folder)
    os.remove(rar_path)

if __name__ == "__main__":
    DATA_FILE_ID = '1oVpVPyH66gN4kL4lQRXcghc-ELgNHdHz'
    DATA_DEST_FOLDER = './artifacts/data'

    download_and_extract(DATA_FILE_ID, DATA_DEST_FOLDER)
