# Stable Deep BSDE Method for Heston Option Pricing using NAIS-Net

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-CQF%20Final%20Project-success)](https://cqf.com)

## Project Overview
This project investigates the implementation of deep neural networks as a computational framework for solving **Backward Stochastic Differential Equations (BSDEs)**. Specifically, it applies this methodology to price European Call options under the **Heston Stochastic Volatility** model.

While traditional numerical methods (Finite Difference) suffer from the "Curse of Dimensionality," and standard Monte Carlo methods struggle with efficient Greek calculation, this Deep BSDE solver leverages the universal approximation capabilities of neural networks to solve the pricing PDE in a scalable manner.

## Key Innovations
This implementation addresses the well-known stability issues of Deep BSDE solvers by integrating two advanced concepts:
1.  **NAIS-Net Architecture:** Utilized the **Non-Autonomous Input-Output Stable Network** (NAIS-Net) architecture. Unlike standard ResNets, NAIS-Net enforces global asymptotic stability constraints on the weight matrices, preventing the "exploding gradient" problem and ensuring robust error propagation through the network.
2.  **Sine Activation Functions:** Experimented with periodic activation functions, demonstrating better capability of modeling the oscillatory nature of gradients compared to traditional Tanh or ReLU functions in specific configurations.

## Mathematical Framework
The project reformulates the Black-Scholes-Heston PDE into a system of Forward-Backward SDEs:

*   **Forward Process (Dynamics):** The stock price $S_t$ and Variance $V_t$ are simulated using an **Euler discretization** scheme, accounting for the correlation $\rho$ between the Brownian motions.
*   **Backward Process (Pricing):** The option price $Y_t$ is modeled as:
    $$dY_t = rY_tdt + Z^S_t dW^S_t + Z^V_t dW^V_t$$
*   **Learning Objective:** The neural network approximates the value function $Y_t = u(t, S_t, V_t)$. Crucially, the sensitivities (Greeks $Z^S$ and $Z^V$) are obtained via **Automatic Differentiation** of the network output, ensuring mathematical consistency between the price and its hedging ratios.

## Implementation Details
*   **State Space:** Inputs $[t, S_t, V_t]$ $\rightarrow$ Output $[Y_t]$.
*   **Loss Function:** A composite loss minimizing the difference between the BSDE forward iteration and the neural network prediction, plus terminal condition matching (Payoff and Delta).
*   **Optimization:** Adam optimizer with a two-phase learning rate schedule (Initial decay followed by fine-tuning).
*   **Regularization Analysis:** Conducted a rigorous study on "Market-Informed Constraints" (No-Arbitrage bounds, Positivity), finding that hard constraints can paradoxically introduce pricing biases (approx. 27% overpricing at inception) compared to unconstrained learning.

## Performance & Results
The model was validated against semi-analytical Heston benchmarks (Fourier transform methods).

*   **Accuracy:** Achieved a Root Mean Square Error (RMSE) as low as **0.0092** (best model configuration).
*   **Batch Size Analysis:** Demonstrated that **single-path training** (Batch Size = 1) yields the lowest RMSE, while larger batch sizes (M=1000) provide smoother convergence and narrower error distributions.
*   **Convergence:** The solver successfully approximates the option value function **along the stochastic trajectories** from $t=0$ to $T$, capturing the complex non-linear relationships of stochastic volatility.

### Visualizations

**Figure 1: Comprehensive Model Performance**
*Top-Right panel demonstrates that the Neural Network predictions (Blue) align with the average Monte Carlo simulation (Green), staying strictly within the 95% confidence intervals. Bottom-Right panel confirms stable convergence of the training loss.*

![Comprehensive Dashboard](dashboard_results.pdf)

**Figure 2: Error Analysis**
*An analysis of Mean Squared Error (MSE) over the option's lifetime reveals that the error peaks at mid-maturity, highlighting the strong guidance provided by the initial and terminal conditions.*

![MSE Analysis](mse_error_analysis.pdf)

## Technologies Used
*   **Language:** Python
*   **Deep Learning:** PyTorch (Automatic Differentiation, Custom Layers for NAIS-Net)
*   **Numerical Computing:** NumPy, SciPy (for Heston semi-analytical benchmarks), pandas (Data management)
*   **Visualization:** Matplotlib

---
## Access to Code
**Due to the strict academic integrity and anti-plagiarism regulations of the Certificate in Quantitative Finance (CQF) program, the source code for this project is hosted in a private repository.**

This repository serves as a portfolio showcase of the methodology and results.
