# Multi-Feature Linear Regression from Scratch in C

Implementation of **multi-feature linear regression in pure C** with no ML libraries—everything from data loading to gradient descent is written manually.

## Overview

Extends single-feature regression to multiple dimensions, focusing on:
- Multi-dimensional gradient computation
- Feature normalization with multiple inputs
- Handling multi-dimensional data in plain C

**Dataset:** [Student performance data](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression) with multiple attributes predicting performance scores.

**Model:**
$$
\text{Score} = w_1 \cdot x_1 + w_2 \cdot x_2 + \cdots + w_n \cdot x_n + b
$$

## Training Details

- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Batch Gradient Descent
- **Preprocessing:** Min-Max normalization on training data
- **Learning rate:** 0.3
- **Split:** 80% train / 20% test (no shuffling)

## Math

**MSE Loss:**
$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( \sum_{j=1}^{n} w_j \cdot x_{i,j} + b - y_i \right)^2
$$

**Gradients:**
$$
\frac{\partial J}{\partial w_j} = \frac{2}{m} \sum_{i=1}^{m} \left( \sum_{k=1}^{n} w_k \cdot x_{i,k} + b - y_i \right) x_{i,j}
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} \left( \sum_{j=1}^{n} w_j \cdot x_{i,j} + b - y_i \right)
$$

**Updates:**
$$
w_j \leftarrow w_j - \alpha \cdot \frac{\partial J}{\partial w_j}, \quad b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}
$$

## Results

```
Loaded 10000 samples with 5 features
Epoch 0 | Train MSE 3413.15
Epoch 500 | Train MSE 9.13
Epoch 1000 | Train MSE 4.49
Epoch 2000 | Train MSE 4.27
Epoch 4900 | Train MSE 4.27

Test MSE: 4.25

Final weights: [2.85, 1.02, 0.62, 0.46, 0.19]
Bias: -33.85

Scikit-Learn comparison:
Weights: [2.85, 1.02, 0.61, 0.48, 0.19]
Bias: -33.92
Test MSE: 4.08
```

## Key Implementation Changes

From single-feature to multi-feature:
- **Input:** 2D array instead of single column
- **Gradients:** Nested loops over features and samples
- **Parameters:** Multiple weights updated per iteration
- **Unchanged:** Core gradient descent algorithm, loss function, normalization approach