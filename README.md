# Neural ODEs for Classification, Generative Modeling, and Trajectory Reconstruction

## 1. Introduction
This project re-implements the key results from the paper "Neural Ordinary Differential Equations" by Chen et al. (2018). The paper introduces Neural ODEs, which reframe neural network depth as a continuous process, offering more efficient computation and memory usage. This project explores the application of Neural ODEs for image classification, generative modeling, and continuous-time trajectory reconstruction.

## 2. Chosen Result
We focus on reproducing three key results from the original paper:
- **ODE-Net Classification**: Neural ODEs are shown to be competitive with ResNet for image classification tasks, specifically MNIST.
- **CNF Two-Moon Transformation**: Demonstrates the use of Continuous Normalizing Flows (CNF) for density estimation.
- **Latent ODE Spiral Reconstruction**: Reconstructs spirals and chirp sinusoids from irregularly sampled data, showcasing the potential of Neural ODEs for continuous-time modeling.

## 3. GitHub Contents
This repository contains the following key directories and files:

- **`code/`**: Contains the implementation of the models.
  - `CNF/`:
    - `CNF.ipynb`: Jupyter notebook for implementing and testing the CNF model.
  - `Resnet/`:
    - `Neural_ODE.ipynb`: Jupyter notebook for implementing and testing the ODE-Net model.
  - `time-series/`:
    - `latent_ode.py`: Python script for implementing the Latent ODE model.
    - `odeint.py`: Script for integration and training of ODE-based models.

- **`data/`**: Contains datasets used for training and evaluation.
  - `CNF/`, `Resnet/`, `time-series/`: Relevant datasets for each model (e.g., `Resnet_data.txt`).
  
- **`results/`**: Includes saved results from the experiments and visualizations.
  - `CNF Paper.pdf`: Paper on CNF.
  - `Resnet_results.pdf`: Results for the ResNet-based ODE-Net.
  - `latent_ode_vis.zip`: Visualizations from the Latent ODE experiments.
  - `learn_physics_figs.zip`, `ode_demos_png.zip`: Additional figures and demos.

- **`poster/`**: Contains a PDF of the project poster.

- **`report/`**: Includes the final project report.
  - `report.pdf`: Full project report detailing the methodology, results, and analysis.

- **`LICENSE.txt`**: License information for the repository.

- **`README.md`**: This documentation file.

## 4. Re-implementation Details
### Models and Datasets:
- **ODE-Net**: Re-implementation for MNIST classification using a continuous-depth neural network.
- **CNF**: Uses a two-layer MLP for Gaussian to two-moon distribution transformation.
- **Latent ODE**: Reconstructs spiral trajectories from irregularly sampled data.

### Tools:
- **PyTorch** for model implementation.
- **Dormand-Prince solver** for CNF training.
- **MNIST**, **Synthetic Spirals**, and **Gaussian datasets** for testing.

### Evaluation Metrics:
- **Classification Accuracy** (ODE-Net).
- **Negative Log-Likelihood** (CNF).
- **Mean Squared Error** (Latent ODE).

### Challenges:
- Adjustments to training steps due to limited computational resources.
- Modifications to solvers for efficiency.

## 5. Reproduction Steps
To re-implement this project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/liandy0127/CS4782-Neural-ODE
   cd CS4782-Neural-ODE
