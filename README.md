# Unsupervised Learning Toolkit: K-Means, GMM, CVAE & More

This repository showcases a collection of essential **unsupervised learning algorithms**, implemented using NumPy and PyTorch. Each method is applied to real-world tasks like **image compression**, **dimensionality reduction**, **source separation**, and **conditional image generation**.

---

## 📌 Overview

| Method | Description |
|--------|-------------|
| K-Means | Hard clustering used for image compression and color quantization. |
| GMM     | Soft probabilistic clustering using the Expectation-Maximization (EM) algorithm. |
| PCA     | Linear dimensionality reduction via eigen decomposition. |
| ICA     | Blind source separation based on statistical independence. |
| CVAE    | Conditional deep generative modeling using Variational Autoencoder architecture. |

---

## 1️⃣ K-Means Clustering

K-Means partitions input data (e.g. image pixeds) into `K` clusters by minimizing intra-cluster variance. It is widely used for **color quantization** and **image compression**.

### Code Demo
```python
def train_kmeans(train_data, initial_centroids, num_iterations=50):
    centroids = initial_centroids.copy()
    for i in range(num_iterations):
        distance = sklearn.metrics.pairwise_distances(centroids, train_data)
        group = np.argmin(distance, axis=0)
        for idx in range(centroids.shape[0]):
            mask = (group == idx)
            if np.any(mask):
                centroids[idx] = np.mean(train_data[mask], axis=0)
    return centroids
```
---
### 📷 Result: K-Means Color Quantization

<img src="Images/Kmeans_Result.png" width="700"/>

---

## 2️⃣ Gaussian Mixture Models (GMM)

GMM extends K-Means by modeling each cluster as a Gaussian distribution, learned via the EM algorithm. It performs **soft assignment**, assigning probabilities to each cluster.

### Code Demo
```python
def train_gmm(train_data, init_pi, init_mu, init_sigma, num_iterations=50):
    pi, mu, sigma = init_pi.copy(), init_mu.copy(), init_sigma.copy()
    for i in range(num_iterations):
        gamma, log_totals = compute_gamma(pi, mu, sigma, train_data)
        num_points = gamma.sum(axis=1, keepdims=True)
        pi = (num_points / len(train_data)).reshape(-1)
        mu = np.matmul(gamma, train_data) / num_points
        for k in range(pi.shape[0]):
            diff = train_data - mu[k]
            sigma[k] = (gamma[k, :, None] * diff).T @ diff / num_points[k]
    return GMMState(pi, mu, sigma)
```
---

### 📷 Result: GMM-Based Image Compression (K=5)

<img src="Images/GMM_Result.png" width="700"/>

---

## 3️⃣ Principal Component Analysis (PCA)

PCA reduces dimensionality by projecting data onto the top eigenvectors of its covariance matrix. It's especially useful for visualizing high-dimensional datasets like **faces** or **gene expressions**.

### Code Demo

```python
def train_PCA(data):
    centered = data - np.mean(data, axis=0)
    cov = np.cov(centered.T, bias=True)
    eig_vals, eig_vecs = np.linalg.eig(cov)
    idx = np.argsort(-eig_vals)
    U = eig_vecs[:, idx]
    eigenvalues = eig_vals[idx]
    return U, eigenvalues
```
---


### 📷 Result: PCA Eigenfaces Visualization

<img src="Images/EigenFace.png" width="700"/>

---

## 4️⃣ Independent Component Analysis (ICA)

ICA separates mixed signals into statistically independent components. Often used in **Audios signunmixes_track_ processing** or **blind source separation**, it differs from PCA by targeting independence rather than uncorrelatedness.

### Code Demo
```python
def unmixer(X):
    W = np.eye(X.shape[1])
    anneal = [0.1]*3 + [0.05]*3 + [0.02]*2 + [0.01]*2
    for alpha in anneal:
        for xi in X:
            W += alpha * (np.outer(1 - 2*sigmoid(W @ xi), xi) + np.linalg.inv(W.T))
    return W

def unmix(X, W):
    return X @ W.T
```
---

### 🎧 Recovered Audio Samples

Click the links below to download or listen to the separated audio signals:

- [🔊 Unmixed Audio 1](Audios/ica_unmixed_track_0.wav)
- [🔊 Unmixed Audio 2](Audios/ica_unmixed_track_1.wav)
- [🔊 Unmixed Audio 3](Audios/ica_unmixed_track_2.wav)
- [🔊 Unmixed Audio 4](Audios/ica_unmixed_track_3.wav)
- [🔊 Unmixed Audio 5](Audios/ica_unmixed_track_4.wav)

> 💡 Note: ICA recovers the signals up to permutation and scaling—so the order and amplitude may not match the true sources, but the content remains clearly separated.

---

## 5️⃣ Conditional Variational Autoencoder (CVAE)

CVAE is a deep generative model that allows you to generate samples **conditioned on a class label**. It is trained on the MNIST dataset to generate digits based on specific labels using the reparameterization trick.

### Code Demo
```python
class CVAE(nn.Module):
    def recognition_model(self, x, c):
        h = self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(torch.cat([x, c], dim=1))))))
        mu = self.layer_mu(h)
        logvar = self.layer_logvar(h)
        return mu, logvar

    def generation_model(self, z, c):
        h = self.relu(self.fc6(self.relu(self.fc5(self.relu(self.fc4(torch.cat([z, c], dim=1))))))
        x_hat = self.sigmoid(self.layer_output(h))
        return x_hat

    def reparametrize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
```

---


### 📷 Result: CVAE Digit Generation (0–9 Grid)

<img src="Images/CAVE_Result.png" width="400"/>

---

## 📄 License

This repository is created for educational purposes (EECS 545 @ University of Michigan). Not licensed for commercial use.

---


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
