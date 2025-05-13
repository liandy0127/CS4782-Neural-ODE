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
     --nsample 300 \
     --subsample 100 \
     --niters 2000
   ```
   ```bash
   python code/time-series/latent_ode.py \
     --dataset sinusoid \
     --nsample 300 \
     --subsample 100 \
     --niters 2000
   ```


## 6 Results/Insights

- **ODE‑Net Classification**  
  - Achieved 98.3 % accuracy on MNIST vs. 98.5 % reported by Chen et al., with only 0.22 M parameters.  
- **CNF Two‑Moon Transformation**  
  - Obtained a negative log‑likelihood of 3.1 nats vs. 0.55 nats in the original work; the learned flow produces a recognizable two‑moon distribution with moderate clustering.  
- **Latent ODE Spiral & Chirp Reconstruction**  
  - ELBO loss improved from below –50 000 to 70 after 2 000 training steps, yielding accurate forward/backward reconstructions of spirals.  
  - Chirp (sinusoidal) trajectories also reconstruct correctly but converge more slowly due to their non‑stationary dynamics.

## 7 Conclusion

- Continuous‑depth ODE‑Nets can match ResNet performance on classification with far fewer parameters.  
- CNF implementations work end‑to‑end but require more training steps or solver‑tolerance tuning to approach original likelihoods.  
- Variational Latent ODEs effectively model irregular time‑series; choice of uniform vs. clustered subsampling has limited impact on final fit.  
- **Lessons Learned:** Sufficient training iterations and optimized ODE‑solver settings are crucial for reproducing high‑fidelity results.

## 8 References

- Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). *Neural Ordinary Differential Equations*. NeurIPS.  
- Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2019). *FFJORD: Free‑Form Continuous Dynamics for Scalable Reversible Generative Models*. ICML.  
- Chen, R. T. Q. et al. (2020). *torchdiffeq* (code repository). https://github.com/rtqichen/torchdiffeq

## 9 Acknowledgements

This work was completed as part of **CS4782: Intro to Deep Learning** at Cornell University. We thank Prof. Kilian Weinberger and Prof. Jennifer Sun for feedback on methodology and presentation, and the open‑source **torchdiffeq** community for the ODE solver implementations.  
```


