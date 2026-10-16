# TDDE01 – Machine Learning

Lab work for the course **TDDE01 Machine Learning** at Linköping University (autumn 2025).
Each lab has three to four assignments: hands-on modelling in R, plus a theory part answered from the course book.

**Tools:** R, R Markdown, `kknn`, `caret`, `glmnet`, `tree`, `ggplot2`, `geosphere`, `kernlab`, `neuralnet`

**Authors:** albal273, lingo978, vikno856. Each lab report has a statement of contributions.

## Repository layout

```
labN/
├── report.Rmd              # combined lab report (knits every assignment + appendix with code)
└── assignmentK/
    ├── assignmentK.Rmd     # write-up for the assignment
    ├── assignmentK.pdf     # rendered report (.html for some)
    ├── R/script.R          # code; pulled into the Rmd via knitr::read_chunk / source
    └── *.csv               # dataset
```

To reproduce an assignment, open its `.Rproj` in RStudio (or set the working directory to the assignment folder) and knit `assignmentK.Rmd`.

---

## Lab 1: Classification and regression basics

| # | Topic | Summary |
|---|-------|---------|
| [1](lab1/assignment1) | **k-nearest neighbours** | Handwritten digit recognition (`optdigits`) with `kknn`. Confusion matrices, heatmaps of easy and hard cases of the digit 8, and selection of *k* from validation misclassification error and cross-entropy. |
| [2](lab1/assignment2) | **Linear and ridge regression** | Predicting Parkinson's `motor_UPDRS` from voice measurements. An `lm` baseline, then ridge regression written by hand (log-likelihood, ridge, and `optim`-based fitting, degrees of freedom) for different values of λ. [PDF](lab1/assignment2/assignment2.pdf) |
| [3](lab1/assignment3) | **Logistic regression** | Diabetes classification (Pima Indians) from glucose and age. Decision boundary, effect of the classification threshold, and basis function expansion. [HTML](lab1/assignment3/assignment3.html) |
| 4 | Theory | Probability thresholds, ways of collecting labels, and the matrix form of the least-squares cost ([`report.Rmd`](lab1/report.Rmd)). |

## Lab 2: Regularisation, trees and PCA

| # | Topic | Summary |
|---|-------|---------|
| [1](lab2/assignment1) | **LASSO and ridge** | Predicting fat content from 100 NIR absorbance channels (Tecator). Shows how plain linear regression overfits, compares LASSO and ridge coefficient paths, and picks λ by cross-validation with `glmnet`. |
| [2](lab2/assignment2) | **Decision trees** | Bank marketing data. Trees with different stopping criteria, pruning to the optimal number of leaves, a loss matrix for imbalanced classes, and ROC (TPR/FPR) compared with logistic regression. [PDF](lab2/assignment2/assignment2.pdf) |
| [3](lab2/assignment3) | **PCA** | Communities and crime data. Hand-made PCA via eigendecomposition and with `princomp`, interpretation of PC1, and linear regression fitted by minimising a custom cost with `optim`. [PDF](lab2/assignment3/assignment3.pdf) |
| 4 | Theory | Reducing the expected new-data error, choosing minibatches, and handling class imbalance ([`report.Rmd`](lab2/report.Rmd)). |

## Lab 3: Kernels, SVMs and neural networks

| # | Topic | Summary |
|---|-------|---------|
| 1 | Theory | The kernel trick ([`report.Rmd`](lab3/report.Rmd)). |
| [2](lab3/assignment2) | **Kernel methods** | Temperature forecast for a location in Sweden, built from Gaussian kernels over physical distance (`geosphere`), day of year and hour. Compares a sum of kernels with a product of kernels. |
| [3](lab3/assignment3) | **Support vector machines** | Spam classification with `kernlab`: model selection for *C*, choosing the final model and the generalisation-error estimate, and reproducing the SVM prediction by hand from the support vectors. [PDF](lab3/assignment3/assignment3.pdf) |
| [4](lab3/assignment4) | **Neural networks** | Learning sin(x) with `neuralnet`. Compares sigmoid, linear, ReLU and softplus activations, tests extrapolation to a wider interval, and tries the inverse problem (x from sin(x)). [PDF](lab3/assignment4/assignment4.pdf) |
