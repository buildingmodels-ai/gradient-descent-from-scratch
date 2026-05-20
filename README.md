# Gradient Descent from Scratch
Manual implementation of gradient descent optimization using NumPy for linear regression.

## Objective
The goal of this project is to understand how gradient descent works by implementing it manually without using machine learning libraries.

## Method
- Generate synthetic linear data: $y = 2x + 1 + \text{noise}$
- Define Mean Squared Error (MSE) loss
- Compute analytical gradients
- Update parameters iteratively
- Visualize loss convergence

## Mathematical Background
For linear regression:

Prediction:
$$
\hat{y} = wX + b
$$

Loss (MSE):
$$
L = \frac{1}{n} \sum (\hat{y} - y)^2
$$

Gradients:
$$
\frac{\partial L}{\partial w} = \frac{2}{n} \sum (\hat{y} - y)X
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{n} \sum (\hat{y} - y)
$$

## Technologies Used
- Python
- NumPy
- Matplotlib

## Result
The algorithm successfully learns parameters close to the true values ($w \approx 2$, $b \approx 1$), demonstrating convergence of gradient descent.

## How to Run
```bash
python gradient_descent.py
```
