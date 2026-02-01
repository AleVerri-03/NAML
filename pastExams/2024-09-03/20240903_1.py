import numpy as np
import matplotlib.pyplot as plt

# import matrix matrix_L.txt
L = np.loadtxt('matrix_L.txt') # Euclidian distance 

# Find P^T*P
# ! Matrice di GramMatrix, quindi P^T*P
G = np.zeros_like(L)
for i in range(L.shape[0]):
    for j in range(L.shape[1]):
        G[i, j] = -0.5 * (L[i, j]**2 - 1/50 * np.sum((L[i, :])**2) - 1/50 * np.sum((L[:, j])**2) + 1/2500 * np.sum(L**2) + 1 / 2500 * np.sum(L**2))

# Use SVD to find P
U, s, VT = np.linalg.svd(G, full_matrices=False)
# Find s^1/2
S_sqrt = np.diag(np.sqrt(s))
# ? Spiegazione: P = Q * S * U^T
# ? P^T = Q^T * S^T * U
# ? So P^T ^ P = U * S^T * Q * Q^T * S * U^T = U * S^2 * U^T -> Note that Q is orthogonal, so Q*Q^T = I
# ? Then we don't knoe Q, so we can't corretlt reconstruct P

# Plot the value of s
plt.close('all')
plt.plot(s, marker='o')
plt.title('Singular values from SVD of Gram Matrix G')
plt.xlabel('Index')
plt.ylabel('Singular value')
plt.yscale('log')
plt.grid()
plt.show()

# Take first 4 columns of U ad transpose
U_4 = U[:, :4].T # Take first 4 rows
# take first 4 singular values
s_4 = S_sqrt[:4, :4]
# Q unitary
Q = np.eye(4)
# Reconstruct P
P = Q @ s_4 @ U_4

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P[0, :], P[1, :], P[2, :], c='b', marker='o')
ax.set_title('3D Reconstruction from Distance Matrix L')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# TODO: DA FINIRE