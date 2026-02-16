# MNIST Digit Classification with MLPs (PyTorch)

Train and analyze linear/shallow/deep MLP baselines on MNIST, with an emphasis on **generalization** (L2 regularization, overfitting curves) and **error analysis** (confusion matrix).

Notebook: `Notebook/ML_project.ipynb`
Report: `Report/Stat_ML_project_report.pdf`

## What’s inside

- Compare three model families (no L2):
  - **Linear model** (no hidden layers)
  - **Shallow MLP** (1 hidden layer)
  - **Deep MLP** (2 hidden layers)
- Study **L2 regularization** on linear models and visualize learned weight “templates” per class (28×28).
- Select the best MLP and evaluate it with:
  - **test accuracy**
  - **confusion matrix built from scratch**
  - discussion of common misclassifications
- Retrain the chosen MLP with different L2 strengths to observe **overfitting** (train vs test loss & accuracy).
- Track **per-class accuracy** over epochs to see which digits are learned faster.

(See the report for full methodology, plots, and discussion.)

## Results (summary)

From the report:
- Best-performing model selected: **Shallow MLP (784 → 512 → 10)**  
- Test accuracy: **98.05%**
- Confusion matrix analysis highlights stronger confusion for digit **9** (e.g., with 3/4/7), and some confusion of **5** as **3**, consistent with visual similarity.
- ![Confusion Matrix](Assets/Confusion_Matrix.png)

Regularization study on the chosen MLP:
- Best L2 setting reported: **λ = 3×10⁻⁵**
- Achieved test loss ≈ **0.048** and test accuracy ≈ **98.5%** (at ~epoch 89), with improved generalization and reduced overfitting compared to no regularization.
- ![Train vs Test Curves over different regularization strengths](Assets/l2test_acc_updated3)

