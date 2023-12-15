# How to run the code 

## Requirements
- Python 3.10.13
- create a conda environment with the following command:
```
conda create -n MedCap python=3.10.13
```
- activate the environment with the following command:
```
conda activate MedCap
``` 

## Install dependencies 
- install the dependencies with the following command:
```
pip install -r requirements.txt
``` 

## Run the code 

- run the code with the following command:
``` 
python main.py
``` 


## since we can not add the data to the deliverables, you can download it from here it will give you a folder named raw, you can place it which you can place it in the directory : artifacts/data/
PS : Make sure to change the name of the folder to raw. So as you can have something like this : artifacts/data/raw 

- Downlaod link :
https://drive.google.com/file/d/1oVpVPyH66gN4kL4lQRXcghc-ELgNHdHz/view?usp=sharing


## run react app for inference
- first u need to install node and npm 
- then u need to install the dependencies with the following command : 
```
sudo apt install nodejs
```
```
sudo apt install npm
```

- first u need to have react 17 installed on your machine in the project directory of webapp/frontend: 
```
npm install react@17 react-dom@17
```

- if you have any other version an you have some problem please run the following code to solve conflicts : npm install --legacy-peer-deps
``` 
npm install --legacy-peer-deps
``` 

- then u need to run the app with the following command in the project directory of webapp/frontend in the terminal: 
```
npm start
``` 

- you should also run the fast api backend using the following command: 
```
python app.py
```