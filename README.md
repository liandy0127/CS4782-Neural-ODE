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

- **`data/`**: Contains datasets used for training and evaluation.
  - `CNF/`, `Resnet/`, `time-series/`: Relevant datasets for each model (e.g., `Resnet_data.txt`).
  
- **`results/`**: Includes saved results from the experiments and visualizations.
  - `CNF Paper.pdf`: Paper on CNF.
  - `Resnet_results.png`: Results for the ResNet-based ODE-Net.
  - `latent_ode_plots.zip`: Visualizations from the Latent ODE experiments.

- **`poster/`**: Contains a PDF of the project poster.

- **`report/`**: Includes the final project report.
  - `report.pdf`: Full project report detailing the methodology, results, and analysis.

- **`LICENSE.txt`**: License information for the repository.

- **`README.md`**: This documentation file.

## 4. Re-implementation Details
### Models and Datasets:
- **ODE-Net**: Re-implementation for MNIST classification using a continuous-depth neural network.
- **CNF**: Uses a two-layer MLP for Gaussian to two-moon distribution transformation.
- **Latent ODE**: Reconstructs spiral and sinusoid trajectories from irregularly sampled data.

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


## 5. Reproduction Steps

1. **Clone the repository**  
   ```bash
   git clone https://github.com/liandy0127/CS4782-Neural-ODE.git
   cd CS4782-Neural-ODE
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   Ensure you have a `requirements.txt` containing:

   ```text
   torch>=1.10.0
   torchdiffeq>=0.2.2
   numpy>=1.20.0
   matplotlib>=3.3.0
   ```

   Then run:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the ODE‑Net MNIST demo**

   ```bash
   jupyter notebook code/Resnet/Neural_ODE.ipynb
   ```

5. **Run the CNF two‑moon demo**

   ```bash
   jupyter notebook code/CNF/CNF.ipynb
   ```

6. **Run the Latent ODE time‑series script**

   ```bash
   python code/time-series/latent_ode.py \
     --dataset spiral \
     --nsample 100 \
     --subsample 50 \
     --clustered True \
     --adjoint True \
     --visualize True \
     --gpu 0
   ```

7. **Inspect results**
   Check the `results/` folder for PNGs and logs.
   To run on the chirp dataset instead:

   ```bash
   python code/time-series/latent_ode.py --dataset chirp --visualize True
   ```

```
```

