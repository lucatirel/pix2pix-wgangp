# WGAN-GP DENOISING

## Prerequisites
Before you begin, make sure you have Python and `git` installed on your system. If you want to use `CUDA`, be sure to have `pytorch` properly installed on your system matching specific `CUDA` version. This project has been tested with Python 3.12.3.

## Installation Instructions

### Step 1: Clone the Repository 
Open your terminal and clone the repository using the command: \
`git clone https://github.com/lucatirel/pix2pix-wgangp`

### Step 2: Set Up the Virtual Environment
After cloning the repository, navigate to the project folder, then create and activate a Python virtual environment:
```
cd pix2pix-wgangp
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```

### Step 3: Install Dependencies
Install the necessary dependencies using pip: 

`pip install -r requirements.txt`

### Step 4: Download and Prepare the Dataset
Download the dataset from the provided URL: 

https://drive.google.com/drive/folders/1FGC1mz4T8WYno-be1YOOrzy6DSVDaKh-?usp=sharing

After downloading, unzip it into the project folder while maintaining the following folder structure: 
```
dataset/
├── clean/
├── noisy/
└── testing/
    ├── clean/
    ├── denoised/
    └── noisy/
```

Ensure the folder structure is exactly as shown above to avoid execution issues.

## Usage

### Training
After following the installation steps, adjust parameters defined in `config/config.json` and start training process with the following command:

`python src/gan_train_pix2pix.py`


### Inference
The script used to perform inference is `src/gan_inference.py`. We provided our best performing pre-trained model in the same url of the dataset mentioned above, named `best_model.pth.tar`. To run inference train a new model or download the pretrained one. Then adjust inference settings in `config/config.json`, and launch inference on the testing set with: 

`python src/gan_inference.py`


