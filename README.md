# Unsupervised Learning Toolkit (EECS 545 HW5)

This repository provides clean and well-documented implementations of fundamental **unsupervised learning algorithms**, applied to tasks like **image compression**, **source separation**, and **generative modeling**. It includes both classical methods implemented in NumPy and deep learning models using PyTorch.

## 📦 Contents

| Algorithm | Description | File(s) |
|----------|-------------|---------|
| **K-Means** | Clustering for image compression. | `kmeans.py`, `kmeans_gmm.ipynb` |
| **Gaussian Mixture Models (GMM)** | EM algorithm for soft clustering and image compression. | `gmm.py` |
| **Principal Component Analysis (PCA)** | Dimensionality reduction using eigen decomposition. | `pca.py`, `pca.ipynb` |
| **Independent Component Analysis (ICA)** | Blind source separation using maximum likelihood. | `ica.py`, `ica.ipynb` |
| **Conditional Variational Autoencoder (CVAE)** | Deep generative model conditioned on labels, applied on MNIST. | `cvae.py`, `cvae.ipynb` |

## 🧪 Requirements

- Python ≥ 3.7
- NumPy
- SciPy
- scikit-learn
- PyTorch
- torchvision
- matplotlib
- Jupyter Notebook

To install the dependencies:

```bash
pip install -r requirements.txt
