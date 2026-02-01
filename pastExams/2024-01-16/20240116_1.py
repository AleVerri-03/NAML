 import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

olivetti = fetch_olivetti_faces()
imgs = olivetti.images
labels = olivetti.target

X = imgs.reshape((400, 4096)).transpose()

# Plot 10 random img and their labels
fig, axs = plt.subplots(ncols=10, nrows=1, figsize=(15, 5))
for i in range(10):
    idx = np.random.randint(0, 400)
    img = imgs[idx, :, :]
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f"Label: {labels[idx]}")
    axs[i].axis("off")
plt.tight_layout()
plt.show()

# Average face
X_avg = np.mean(X, axis=1, keepdims=True)
fig = plt.figure(figsize=(6, 6))
plt.imshow(X_avg.reshape((64, 64)), cmap='gray')
plt.title("Average Face")
plt.axis("off")
plt.show()

# Compute SVD full and reduced
import time
start_full = time.time()
U_full, S_full, VT_full = np.linalg.svd(X, full_matrices=True)
end_full = time.time()
print(f"Full SVD time: {end_full - start_full:.4f} seconds")
start_reduced = time.time()
U_red, S_red, VT_red = np.linalg.svd(X, full_matrices=False)
end_reduced = time.time()
print(f"Reduced SVD time: {end_reduced - start_reduced:.4f} seconds")

# ! In the truncated version we save space by rapresenting only
# ! the first min(m,n) columns of U and rows of V^T
# ! Both versione are exact decompositions of the original matrix X

# Plot trand of singular values, y axis in log scale
plt.figure(figsize=(8, 5))
plt.plot(S_full, marker='o')
plt.title("Singular Values Trend")
plt.xlabel("Index")
plt.ylabel("Singular Value") # ! y 
plt.yscale('log')
plt.grid()
plt.show()

# Explained variance
explained_variance = (S_full ** 2) / np.sum(S_full ** 2)
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid()
plt.show()

# SVD rank k approximation

def randomized_SVD(A, k):
    _, n = A.shape
    P = np.random.randn(n, k)
    Z = A @ P
    Q, _ = np.linalg.qr(Z)
    Y = Q.T @ A
    Uy, sy, VTy = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy
    return U, sy, VTy

k = [1, 5, 10, 50, 100]
# Plot original and approximated singular values
fig, axs = plt.subplots(1, len(k)+1, figsize=(20, 5))
axs[0].plot(S_full, marker='o')
axs[0].set_title("Original Singular Values")
for i, ki in enumerate(k):
    U_k, S_k, VT_k = randomized_SVD(X, ki)
    axs[i+1].plot(S_k, marker='o')
    axs[i+1].set_title(f"Randomized SVD k={ki}")
plt.tight_layout()
plt.show()

# PCA to perform dimensionality reduction
# rank k approximation
X_norm = X - X_avg
U, S, VT = np.linalg.svd(X_norm, full_matrices=False)

dim = X.shape[0] * X.shape[1]
error = list()
# Calculate explained variance
for ki in k:
    X_k = U[:, :ki] @ np.diag(S[:ki]) @ VT[:ki, :]
    explained_var = np.linalg.norm(X_norm - X_k, 'fro')**2 / dim
    # ! Norm Frobenius squared: ||X_origin - X_k||_F^2 / (m*n)
    error.append(explained_var)
    print(f"Explained variance with k={ki}: {explained_var:.4f}")

# Plot explained variance by k
plt.figure(figsize=(8, 5))
plt.plot(k, error, marker='o')
plt.title("Explained Variance by k")
plt.xlabel("k")
plt.ylabel("Explained Variance")
plt.grid()
plt.show()

# Plot first 30 axis of U as images
fig, axs = plt.subplots(3, 10, figsize=(15, 5))
for i in range(30):
    ax = axs[i // 10, i % 10]
    ax.imshow(U[:, i].reshape((64, 64)), cmap='gray')
    ax.set_title(f"U axis {i+1}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# Select label 0 or 39
zero_label = np.where(labels == 0)[0]
thirty_nine_label = np.where(labels == 39)[0]

# X has shape (pixels, faces). Labels refer to faces.
# To select faces, index the columns of X.
# Concatenate the selected faces along axis=1 to stack them as columns.
X_small = np.concatenate((X[:, zero_label], X[:, thirty_nine_label]), axis=1)

# Normalize X_small. Subtract the mean of each row (pixel) from all columns (faces).
# np.mean(X_small, axis=1) will have shape (num_pixels,).
# To enable broadcasting, reshape it to (num_pixels, 1) using keepdims=True.
X_small_norm = X_small - np.mean(X_small, axis=1, keepdims=True)
U_small, S_small, VT_small = np.linalg.svd(X_small_norm, full_matrices=False)

# Plot first 2 components
figure, ax = plt.subplots(1, 2, figsize=(10,20))
for i in range(2):
    ax[i].imshow(U_small[:, i].reshape((64, 64)), cmap='gray')
plt.show()

# 10. Create a scatterplot for the first 2 principal components of the subset of images grouped by label. Comment what you see.

# Calculate principal components for label 0 faces
num_zero_faces = zero_label.shape[0]
X_small_zero = X_small[:, :num_zero_faces]
x_zero = X_small_zero.T @ U_small[:, 0]
y_zero = X_small_zero.T @ U_small[:, 1]

# Calculate principal components for label 39 faces
X_small_thirty_nine = X_small[:, num_zero_faces:]
x_thirty_nine = X_small_thirty_nine.T @ U_small[:, 0]
y_thirty_nine = X_small_thirty_nine.T @ U_small[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(x_zero, y_zero, color='r', label='Label 0')
plt.scatter(x_thirty_nine, y_thirty_nine, color='b', label='Label 39')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatterplot of First 2 Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# Loss Function ofr the ridge regression
# L(w) = ||y - Xw||^2 + lambda||w||^2

import jax.numpy as jnp
def ridge_loss(w, X, y, lambd):
    prevision = jnp.dot(X, w)
    loss = jnp.sum((y - prevision) ** 2) + lambd * jnp.sum(w ** 2)
    return loss

np.random.seed(55)
x = np.arange(np.pi,3*np.pi,0.1)
y = np.sin(x) + np.random.normal(0,0.1,len(x))

degree = 1 # ! Degree of the polynomial
X = np.vander(x, N=degree+1, increasing=True) # ! Vandermonde matrix

lambda_ = [0, 10**-32, 10**-16, 10**-8, 1, 16, 32, 1024]

w = list()
for lambd in lambda_:
    # ! Posso ignorare i coefficienti costanti
    tmp = np.linalg.inv(X.T @ X + lambd * np.eye(X.shape[1])) @ (X.T @ y)
    w.append(tmp)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='black', label='Data Points')
x_plot = jnp.linspace(np.pi, 3*np.pi, 1000)
for i, lambd in enumerate(lambda_):
    y_plot = jnp.dot(jnp.vander(x_plot, N=degree+1, increasing=True), w[i]) # ! Plot del vettore
    plt.plot(x_plot, y_plot, label=f'Lambda: {lambd}')
plt.title('Ridge Regression with Different Lambda Values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

# Commenti:
# For versy small lambda values, like 10^-16 and above, the model
# can't find the solution, so the linear regression doesn't fit the data.
# for lambda from 10^-8 the model starts to fit the data well, when lambda
#increases the model starts to be flatter, like the casse lambda = 1024