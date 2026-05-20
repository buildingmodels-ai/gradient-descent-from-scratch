# Gradient Descent from Scratch
Manual implementation of gradient descent optimization using NumPy for linear regression.

## Objective
The goal of this project is to understand how gradient descent works by implementing it manually without using machine learning libraries.

## Method
- Generate synthetic linear data: y = 2x + 1 + noise
- Define Mean Squared Error (MSE) loss
- Compute analytical gradients
- Update parameters iteratively
- Visualize loss convergence

## Mathematical Background
For linear regression:

Prediction:
ŷ = wX + b

Loss (MSE):
L = (1/n) Σ (ŷ - y)²

Gradients:
∂L/∂w = (2/n) Σ (ŷ - y)X  
∂L/∂b = (2/n) Σ (ŷ - y)

## Technologies Used
- Python
- NumPy
- Matplotlib

## Result
The algorithm successfully learns parameters close to the true values (w ≈ 2, b ≈ 1), demonstrating convergence of gradient descent.

## How to Run
```bash
python gradient_descent.py
```
