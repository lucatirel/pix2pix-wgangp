# WGAN-GP DENOISING

## Prerequisites
Before you begin, make sure you have Python and `git` installed on your system. This project has been tested with Python 3.12.3

## Installation Instructions

### Step 1: Clone the Repository 
Open your terminal and clone the repository using the command: \
'''git clone https://github.com/lucatirel/pix2pix-wgangp'''

### Step 2: Set Up the Virtual Environment
After cloning the repository, navigate to the project folder:
cd project_folder_name
Create and activate a Python virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

### Step 3: Install Dependencies
Install the necessary dependencies using `pip`:
pip install -r requirements.txt

### Step 4: Download and Prepare the Dataset
Download the dataset from the provided URL:
https://drive.google.com/drive/folders/1FGC1mz4T8WYno-be1YOOrzy6DSVDaKh-?usp=sharing
After downloading, unzip it into the project folder while maintaining the following folder structure:
dataset/
├── clean/
├── noisy/
└── testing/
    ├── clean/
    ├── denoised/
    └── noisy/
Ensure the folder structure is exactly as shown above to avoid execution issues.

## Usage
After following the installation steps, adjust parameters defined in src/gan_train_pix2pix.py and launch training process with the following command:
python src/gan_train_pix2pix.py


