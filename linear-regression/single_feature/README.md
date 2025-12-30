# Linear Regression from Scratch in C

This project implements **single-feature linear regression from scratch in C**.
No machine learning libraries are used — everything from data loading to gradient
descent is written manually to understand how the algorithm actually works.

---

## Why this project?

Most machine learning tutorials hide the core mechanics behind high-level libraries.
This project focuses on understanding:

- where the loss function comes from
- how gradients are derived
- why feature normalization matters
- how gradient descent behaves numerically

Writing the model in C forces every step to be explicit.

---

## Why this dataset?

The SAT–GPA dataset was chosen because:

- it is **simple and low-dimensional** (one feature, one target)
- the relationship is approximately linear
- it is easy to visualize and reason about
- it is commonly used as a beginner example

---

## Problem statement

Given SAT scores, predict GPA using a linear model:

$$
\text{GPA} = w \cdot \text{SAT} + b
$$

where:

- $w$ is the slope (weight)
- $b$ is the intercept (bias)

---

## Dataset

- File: [`sat-gpa.csv`](data/sat-gpa.csv)
- Source: [Kaggle](https://www.kaggle.com/datasets/luddarell/101-simple-linear-regressioncsv)
- Feature: SAT score
- Target: GPA
- Split: 75% training, 25% testing
- Data is split without shuffling

---

## Training method

- Loss function: **Mean Squared Error (MSE)**
- Optimization: **Batch Gradient Descent**
- Feature normalization: **Min–Max scaling (training data only)**
- Adaptive learning rate (reduced if loss increases)
- Early stopping based on loss convergence

---

## Where the equations come from

For a single training example $(x_i, y_i)$, the model prediction is:

$$
\hat{y}_i = w \cdot x_i + b
$$

The error (residual) for that sample is the difference between the prediction
and the true value:

$$
e_i = \hat{y}_i - y_i
$$

Substituting the model equation:

$$
e_i = w \cdot x_i + b - y_i
$$

To measure how well the model fits all training samples, the **Mean Squared Error**
(MSE) loss function is used:

$$
J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \left( w \cdot x_i + b - y_i \right)^2
$$

where:

- $m$ is the number of training samples

To minimize this loss, gradient descent is applied by computing the partial
derivatives of $J(w, b)$ with respect to each parameter.

The gradient with respect to the weight $w$ is:

$$
\frac{\partial J}{\partial w} = \frac{2}{m} \sum_{i=1}^{m} \left( w \cdot x_i + b - y_i \right) x_i
$$

The gradient with respect to the bias $b$ is:

$$
\frac{\partial J}{\partial b} = \frac{2}{m} \sum_{i=1}^{m} \left( w \cdot x_i + b - y_i \right)
$$

During training, the parameters are updated by moving in the opposite direction
of the gradient:

$$
w \leftarrow w - \alpha \cdot \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \alpha \cdot \frac{\partial J}{\partial b}
$$

where $\alpha$ is the learning rate.

All of these equations are implemented directly in the C code.

---

## Results

After training, the learned parameters are converted back to the original
SAT scale and evaluated on the test set.

Example model form:

$$
\text{GPA} = 0.001326 \times \text{SAT} + 0.824499
$$

(The exact values depend on the dataset.)

---

## Visualization

### Data distribution

Distribution of SAT and GPA values to inspect the spread of the data.

![Data distribution](linear-regression/single_featureplots/plots/data_distribution.png)

### Linear regression fit

Scatter plot of SAT vs GPA with the learned regression line overlaid.

![Regression fit](linear-regression/single_featureplots/plots/regression_fit.png)

### Loss curve

Mean Squared Error plotted over training iterations, showing gradient descent
convergence.

![Loss curve](linear-regression/single_featureplots/plots/loss_curve.png)

### Comparison with scikit-learn

Predictions from the from-scratch C implementation compared with a
single-feature linear regression model from scikit-learn.

![Comparison with scikit-learn](linear-regression/single_featureplots/plots/comparison_with_scikit-learn.png)

---

## Future work

- Add data shuffling before train/test splitting for more robust evaluation
- Extend the implementation to multi-feature linear regression

---

## Conclusion

This project shows that linear regression is simply an optimization problem over a convex loss surface.
