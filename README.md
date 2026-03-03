# **K-Nearest Neighbors Classifier**

A full KNN implementation from scratch in Python, including:

* Efficient vectorized KNN classifier
* Data sampling utilities
* Experiments on MNIST
* Error analysis across sample sizes, k-values, and corrupted labels
* Visual plots of performance

This project focuses on implementing and analyzing the K-Nearest Neighbors algorithm **without relying on scikit-learn**.

---

## **📌 Project Overview**

This repository implements a complete workflow for evaluating KNN using subsets of the MNIST digit dataset (digits **1, 3, 4, 6**):

### **Key Components**

* **Custom KNN implementation**
* **Optimized vectorized distance computation**
* **Random training set sampling**
* **Experiments measuring test error vs:**

  * training sample size (Q2a)
  * k-value (Q2d)
  * robustness under 30% label corruption (Q2e)
* **Matplotlib visualizations**

The entire pipeline is designed to be self-contained and educational.

---

## **📂 Repository Structure**

```
├── nearest_neighbour.py.py    # Contains all KNN functions and experiment code
├── mnist_all.npz              # MNIST dataset (should be placed in project root)
└── README.md
```

---

## **⚙️ Core Functionality**

### **1. `gensmallm` — Random Sampling**

Randomly samples `m` training examples across selected digit classes, preserving labels.

### **2. `learnknn` — Train KNN**

Stores:

* number of neighbors `k`
* training features
* training labels

### **3. `predictknn` — Vectorized Prediction**

Performs fast classification using:

[
||a - b||^2 = ||a||^2 + ||b||^2 - 2a \cdot b
]

This avoids explicit loops and computes a full distance matrix efficiently.

### **4. `Q2a`, `Q2d`, `Q2e` Experiments**

Each experiment:

* samples data
* trains KNN
* predicts on MNIST test set
* computes average error over multiple runs
* plots the results

**Q2e** additionally corrupts 30% of labels to evaluate robustness.

---

## **📊 Experiment Summaries**

### **Q2a — Error vs Training Size (`m`)**

Tests KNN with **k = 1** and training sizes:

```
m ∈ {1, 10, 20, 30, ..., 100}
```

Plots:

* average error
* min–max error bars

---

### **Q2d — Error vs k (Clean Labels)**

For each `m ∈ {50, 150, 500}`, evaluates:

```
k = 1 ... 15
```

Shows how larger training sets affect the optimal choice of k.

---

### **Q2e — Error vs k (30% Corrupted Labels)**

Same as Q2d, but after intentionally corrupting 30% of labels in:

* training data
* test data

This highlights how KNN behaves under noisy labeling.

---

## **🔬 Label Corruption Function (Q2e)**

Labels are corrupted by replacing them with a random **wrong** label among the valid digit classes:

```python
valid_labels = np.array([1, 3, 4, 6])
labels[corrupted_indices] = [
    np.random.choice(valid_labels[valid_labels != current_labels])
    for current_labels in labels[corrupted_indices]
]
```

---

## **🚀 How to Run**

1. Place `mnist_all.npz` in the project directory
2. Run individual experiments:

```bash
python main.py
```

Inside the file, uncomment only the experiment you want:

```python
simple_test()
Q2a_code()
Q2d_code()
Q2e_code()
```

Plots will automatically display.

---

## **🛠️ Technologies Used**

* **Python**
* **NumPy**
* **Matplotlib**
* **SciPy (distance utilities)**

No machine-learning libraries are used — all logic is implemented manually.

---

## ** In a Nutshell**

**What was Implemented:**

* Full KNN algorithm from scratch
* Vectorized distance calculations for significant speed improvements
* Training/test sampling utilities
* All experiment loops (Q2a, Q2d, Q2e)
* Plotting and performance analysis
* Label corruption mechanism for noise robustness study

This project demonstrates of:
✔ KNN internals
✔ Bias–variance tradeoff
✔ Efficiency in numerical Python
✔ Experimental methodology in ML
✔ Working with real datasets

---

## **📈 Example Plots**

Plots generated during the experiment (examples):

* Error vs. training size (Q2a)
(Add images here)

* Error vs. k for multiple sample sizes (Q2d)
(Add images here)


* Error vs. k under noise (Q2e)
(Add images here)

---

## **📜 License**

This project is for educational and portfolio purposes.

---
