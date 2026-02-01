import scipy.io as sio
data = sio.loadmat('faces.mat')
X = data['X']
# Collection 5000: 32x32

# Plot single face
import numpy as np
import matplotlib.pyplot as plt
x0 = np.transpose(np.reshape(X[0,:], (32,32))) # ! Photo x row-wise
# ! So U contains Eigenfaces, VT contains coefficients for each face
# plt.imshow(x0, cmap='gray')
# plt.show()

# 1. Normalized X
X_mean = np.mean(X, axis=0)

X_bar = X - X_mean[None, :] # Centered data matrix

# print mean face (try)
x_mean = np.transpose(np.reshape(X_mean[:], (32,32)))
# plt.imshow(x_mean, cmap='gray')
# plt.show()

# PCA on 25 eigenfaces
n_eigenfaces = 25
# ! Transposte X in order to use the SVD as in the lecture
X_SVD = X_bar.T
U, s, VT = np.linalg.svd(X_SVD, full_matrices=False)
# U: eigenfaces, s: singular values, VT: coefficients for each face

Phi = U.T @ X_bar.T # ! Project data onto principal components
# Phi: coefficients of each face in the eigenface basis
# Plot first 25 eigenfaces
plt.close('all')
fig, axs = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
axs = axs.reshape((-1,))
for i in range(n_eigenfaces):
    eigenface_i = np.transpose(U[:, i].reshape((32, 32)))
    axs[i].imshow(eigenface_i, cmap="gray")
    axs[i].set_title(f"$u_{{{i + 1}}}$")
    axs[i].axis("off")
# plt.show()

# Sample dimension from 32x32 to 10x10
U_100 = U[:, :100]
Phi_100 = U_100.T @ X_bar.T
X_1 = Phi_100[:, 0].reshape((10, 10)) # ! Reshape the first face to 10x10
X_2 = Phi_100[:, 1].reshape((10, 10)) # ! Reshape the second face to 10x10
plt.close('all')
fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

x_1 = np.transpose(X_1)
x_2 = np.transpose(X_2)
ax[0].imshow(x_1, cmap='gray')
ax[1].imshow(x_2, cmap='gray')
plt.show()

# plot first 100: original, reconstructed, absolute error
plt.close('all')
fig, axs = plt.subplots(nrows=100, ncols=3, figsize=(15, 15))
axs = axs.flatten()  # Flatten the 2D array to 1D for easy indexing
for i in range(100):
    # original
    axs[i].imshow(np.transpose(X[i, :].reshape((32, 32))), cmap='gray')
    axs[i].axis('off')
    # reconstructed
    reconstructed_i = X_mean + U_100 @ Phi_100[:, i]
    axs[i + 100].imshow(np.transpose(reconstructed_i.reshape((32, 32))), cmap='gray')
    axs[i + 100].axis('off')
    # absolute error
    error_i = np.abs(X[i, :] - reconstructed_i)
    axs[i + 200].imshow(np.transpose(error_i.reshape((32, 32))), cmap='gray')
    axs[i + 200].axis('off')

plt.show()
