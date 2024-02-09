# Machine-Learning: Supervised Learning Techniques

## Overview

This repository contains a Jupyter Notebook (`SL_Assigment1.ipynb`) that demonstrates the application of various supervised learning techniques on two different datasets: Wine Quality Dataset and Breast Cancer Dataset. The notebook explores decision tree, neural network, boosting, SVM and K-neighbour classifiers for both datasets, showcasing learning curves, validation curves, and hyperparameter tuning.

## Dependencies

Before running the notebook, ensure you have the necessary dependencies installed. You can install them using the following command:

```bash
pip install scikit-learn
```
```bash
pip install pandas
```
```bash
pip install numpy
```
```bash
pip install matplotlib
```

## Datasets

1. **Wine Quality Dataset:**
   - Dataset file: `drive/MyDrive/winequality-white.csv`
   - Features (`X_wine`): All columns except the last one
   - Labels (`y_wine`): Quality Group (categorized as 'Bad' or 'Good')

2. **Breast Cancer Dataset:**
   - Dataset file: `drive/MyDrive/wdbc.data`
   - Features (`X_cancer`): All columns except the first one
   - Labels (`y_cancer`): 'M' (Malignant) or 'B' (Benign)

## Usage

1. Open the Jupyter Notebook `SL_Assigment1` in a compatible environment (e.g., Google Colab).
2. Adjust the file paths for the datasets (`file_path_wine` and `file_path_cancer`) if needed.
3. Run the cells sequentially to execute the code and observe the results.

## Decision Tree Classifier

### Wine Quality Dataset
- Function: `decision_tree(X, y, depth, splitter, scorer)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='Good')
  decision_tree(X_wine, y_wine, 15, 'best', scorer)
  ```

### Breast Cancer Dataset
- Function: `decision_tree(X, y, depth, splitter, scorer)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='B')
  decision_tree(X_cancer, y_cancer, 3, 'random', scorer)
  ```

## Neural Network Classifier

### Wine Quality Dataset
- Function: `neural_network(X, y, scorer, size, activation)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='Good')
  neural_network(X_wine, y_wine, scorer, (10, 5), 'logistic')
  ```

### Breast Cancer Dataset
- Function: `neural_network(X, y, scorer, size, activation)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='B')
  neural_network(X_cancer, y_cancer, scorer, (15,), 'relu')
  ```

## AdaBoosting Classifier

### Wine Quality Dataset
- Function: `adaboost_clasifier(X, y, scorer, n_est, learn_rate)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='Good')
  adaboost_clasifier(X_wine,y_wine,scorer,50,0.5)
  ```

### Breast Cancer Dataset
- Function: `adaboost_clasifier(X, y, scorer, n_est, learn_rate)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='B')
  adaboost_clasifier(X_cancer,y_cancer,scorer,80,1.25)
  ```

## SVM Classifier

### Wine Quality Dataset
- Function: `svc(X,y,scorer,c)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='Good')
  svc(X_wine,y_wine,scorer,3)
  ```

### Breast Cancer Dataset
- Function: `svc(X,y,scorer,c)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='B')
  svc(X_cancer,y_cancer,scorer,9)
  ```

## K-neighbours Classifier

### Wine Quality Dataset
- Function: `k_neighbors(X,y,scorer,neighbours)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='Good')
  k_neighbors(X_wine,y_wine,scorer,10)
  ```

### Breast Cancer Dataset
- Function: `k_neighbors(X,y,scorer,neighbours)`
- Example:
  ```python
  scorer = make_scorer(f1_score, pos_label='B')
  k_neighbors(X_cancer,y_cancer,scorer,3)
  ```
