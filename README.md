## Introduction

\# User Guide for a New 2D Magnetotelluric Inversion Code

&nbsp;   This project implements a 2D magnetotelluric (MT) resistivity inversion network based on a selective state-space model (SSM) and provides Transformer and U-Net networks for comparison.



\## Overview

&nbsp;    The codebase is organized into three main components:

&nbsp;    --Data Generation: Synthetic MT data generation scripts

&nbsp;    --Model Training: Implementation of multiple inversion approaches

&nbsp;    --Model Prediction: Inference and evaluation scripts



\### Implemented Methods

&nbsp;     This includes testing of various theoretical 2D resistivity models.

      --MambaMT2DINV: Our proposed inversion method based on SSM

      --Transformer: Transformer network for 2D MT inversion

      --U-Net: architecture for comparative analysis

&nbsp;     Extensive testing has been conducted on various theoretical resistivity models to validate all methods.



\#### Prerequisites

      Required Software

&nbsp;     --Python 3.12 with the following packages:

           --PyTorch, NumPy, SciPy, Matplotlib, h5py

      --CUDA -enable GPU (This experiment tested on NVIDIA GeForce RTX 4080 SUPER GPU)

      --MATLAB 2023a (for LSM inversion and certain plotting functions)

      --Tecplot (for 2D result visualization)

      --Surfer (for geophysical data visualization)



\##### Installation

&nbsp;     --1. Clone the repository: 

&nbsp;          --git clone https://github.com/ZhaoZiYan1/MT2DINV
&nbsp;          --cd MT2DINV

&nbsp;     --2. Install Python dependencies (if using pip):

&nbsp;          --pip install NumPy

&nbsp;          --pip install Matplotlib

           --pip install Torch

           --pip install Scipy



\###### Quick Test for Reviewers

&nbsp;     To verify installation and basic functionality, run one of the test scripts:

&nbsp;     --Test different inversion scenarios

&nbsp;     --1.Model Training and Prediction

&nbsp;         python Train/Mamba_MT2D_Train.py              # Train the proposed Mamba-based 2D MT inversion network

&nbsp;         python Train/Mamba_MT2D_Predicted.py          # Predict 2D resistivity models using the trained Mamba network

&nbsp;         python Train/Mamba_MT2D_ModeTrain.py          # Train the Mamba network using different input modes (TE, TM, and TE+TM)

&nbsp;         python Train/Mamba_MT2D_ModePredicted.py      # Evaluate inversion performance under different input modes

&nbsp;         python Train/Mamba_MT2D_NoisyTrain.py         # Train the Mamba network using noisy MT data

&nbsp;         python Train/Mamba_MT2D_NoisyPredicted.py     # Evaluate inversion performance under noisy conditions


&nbsp;         python Train/Trans_MT2D_Train.py              # Train the Transformer network for comparative analysis


&nbsp;         python Train/Trans_MT2D_Predicted.py          # Predict 2D resistivity models using the Transformer network

&nbsp;         python Train/UNet_MT2D_Train.py               # Train the U-Net network for comparative analysis

&nbsp;         python Train/UNet_MT2D_Predicted.py           # Predict 2D resistivity models using the U-Net network

&nbsp;     --2. Network Architecture and Utility Modules

&nbsp;         python Train/Mamba.py                         # Core implementation of the selective state-space model (Mamba)

&nbsp;         python Train/Mamba_MT2D_Inversion.py          # Architecture of the proposed Mamba-based 2D MT inversion network

&nbsp;         python Train/Trans_MT2D_Inversion.py          # Architecture of the Transformer-based 2D MT inversion network

&nbsp;         python Train/UNet_MT2D_Inversion.py           # Architecture of the U-Net-based 2D MT inversion network

&nbsp;         python Train/Utilities.py                     # Data loading, preprocessing, and normalization utilities

&nbsp;     --3. Inversion Results and Statistical Analysis

&nbsp;         python Train/SSIM_Distribution_Plot.py        # Compare SSIM distributions of the Mamba, Transformer, and U-Net networks

&nbsp;         python Train/Export_SSIM_Advantage_Excel.py   # Export ranked SSIM advantages of Mamba over the comparison networks to Excel

&nbsp;     --4. Synthetic Dataset Generation and Forward Modeling

&nbsp;         python Training_dataset/Gaussian_Random_Fields.py       # Generate Gaussian random fields for constructing 2D resistivity models

&nbsp;         python Training_dataset/Validate_Classic_1D_Models.py   # Validate 2D MT forward modeling against analytical solutions for 1D layered models

&nbsp;         python Training_dataset/Generate_Noisy_Dataset.py       # Generate noisy TM-mode datasets for robustness experiments

&nbsp;         python Training_dataset/Model_Plot.py                   # Visualize 2D resistivity models and corresponding MT responses

&nbsp;     --5. Forward Modeling Modules

&nbsp;         python Training_dataset/MT2D_FDM.py      # Finite-difference solver for 2D MT forward modeling in TE and TM modes

&nbsp;         python Training_dataset/MT2D_Model.py    # 2D MT forward modeling module with parallel computation support

&nbsp;     --6. MATLAB Visualization

&nbsp;         run('Training_dataset/MT_Validation_1DForward.m')   % Compare 1D analytical and 2D FDM responses for H-type and K-type models

&nbsp;         run('Training_dataset/Draw_2D_ResistivityModel.m')  % Export 2D resistivity models to Tecplot format



\####### Project Structure

&nbsp; MT2DINV/

├── 1D/                                          # 1D inversion implementations

│   ├── FCNN/                                # Fully Connected Neural Network implementation



\######## Notes

&nbsp;     --The Mamba-based 2D MT inversion method is the primary focus of this repository.

&nbsp;     --Transformer and U-Net networks are provided for comparative analysis.

&nbsp;     --The code includes inversion experiments on various theoretical 2D resistivity models, noise robustness tests, and comparisons of different input modes (TE, TM, and TE+TM).

&nbsp;     --The synthetic datasets are generated using Gaussian random fields and 2D MT finite-difference forward modeling.

&nbsp;     --MATLAB is required for forward modeling validation and certain visualization tasks.

&nbsp;     --CUDA-enabled GPUs are recommended for efficient network training and prediction.

&nbsp;     --Training datasets and pre-trained model parameters are required to reproduce the inversion experiments.



\######### Contact

&nbsp;     --For questions or support, please contact: E-mail：18403309506@163.com



\########## License

&nbsp;     --This project is for research purposes. Please cite our work if you use this code in your research.
