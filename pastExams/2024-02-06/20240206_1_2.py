import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# X 400x400 with
# = 110 50<=i<=150 30<=j<=130
# = 150 50<=i<=150 230<=j<=330 
# = 180 250<=i<=350 70<=j<=170
# = 220 250<=i<=350 270<=j<370
# = 0 elsewhere
X = np.zeros((400, 400))
X[49:150, 29:130] = 110
X[49:150, 229:330] = 150
X[249:350, 69:170] = 180
X[249:350, 269:370] = 220

# Add noise
np.random.seed(0)
noise = np.random.randn(400, 400) * 0.2
X_noisy = X + noise

def SVT(X_noisy, X, threshold, mask_keep, n_max_iter=100, increment_tol=1e-6): 
    errors = list()
    A = X_noisy.copy()
    
    for i in range(n_max_iter):
        A_old = A.copy()

        U,s,VT = np.linalg.svd(A, full_matrices = False)
        s[s < threshold] = 0

        A = U @ np.diag(s) @ VT  
        A[mask_keep] = X[mask_keep]

        increment = np.linalg.norm(A - A_old)
        errors.append(np.linalg.norm(X-A)/np.linalg.norm(X))

        if np.linalg.norm(A)==0:
            print('k=', threshold, 'removed all singular values')
            return A, errors
        
        if increment < increment_tol:
            break
    return A, errors

reconstruction = list()
error = list()
thresholds = [0.1, 1, 5, 10, 20]
mask = np.zeros_like(X, dtype=bool) # ! come fare una matrice di false
for th in thresholds:
    X_rec, err = SVT(X_noisy, X, th, mask)
    reconstruction.append(X_rec)
    error.append(err)

# Plot results
plt.close('all')
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axs = axs.flatten()
axs[0].imshow(X_noisy, cmap='gray')
axs[0].set_title('Noisy Image')
for i in range(len(thresholds)):
    axs[i+1].imshow(reconstruction[i], cmap='gray')
    axs[i+1].set_title(f'Reconstruction (th={thresholds[i]})')
plt.show()

# Plot error convergence for each threshold
plt.close('all')
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(thresholds)):
    ax.plot(error[i], label=f'th={thresholds[i]}')
ax.set_title('Error Convergence')
ax.set_xlabel('Iteration')
ax.set_ylabel('Relative Error')
ax.legend()
plt.show()

def randomized_SVD(A, k):
  n, m = A.shape[0], A.shape[1]
  P = np.random.randn(m, k)
  Z = A @ P
  Q, R = np.linalg.qr(Z)
  Y = Q.T @ A
  U_Y, s, VT = np.linalg.svd(Y, full_matrices = False)
  U = Q @ U_Y
  return U, s, VT

k_r = [50, 5, 3, 2, 1]
np.random.seed(42)

#plot
plt.close('all')
fig, axs = plt.subplots(nrows=1, ncols=len(k_r), figsize=(15, 5))
for i, k in enumerate(k_r):
    U_r, s_r, VT_r = randomized_SVD(X_noisy, k)
    X_r = U_r @ np.diag(s_r) @ VT_r
    axs[i].imshow(X_r, cmap='gray')
    axs[i].set_title(f'Randomized SVD (k={k})')
plt.show()

# ! Characterize Stationary Points

import sympy as sym
import jax
import jax.numpy as jnp

x_sym = sym.Symbol('x_sym', real = True)
y_sym = sym.Symbol('y_sym', real = True)
f_sym = 2*x_sym**2 - 1.05*x_sym**4 + x_sym**6/6 + x_sym*y_sym + y_sym**2 # ? Write the function

print('gradient is: ')
gradient = sym.Matrix([f_sym]).jacobian([x_sym, y_sym])
gradient

stationary_points = sym.solve(gradient, (x_sym, y_sym))
print('Stationary points are: ', stationary_points)

# Calcola la matrice Hessiana simbolicamente
H_sym = sym.hessian(f_sym, (x_sym, y_sym))

# Lista per memorizzare il tipo di ciascun punto stazionario
points_type = []
for point in stationary_points:
    # Sostituisci i valori del punto nella matrice Hessiana
    hessian_cur = H_sym.subs({x_sym: point[0], y_sym: point[1]})
    
    # Calcola gli autovalori della Hessiana in quel punto
    # eigenvals() ritorna un dizionario {valore: molteplicità}
    eigenvals_dict = hessian_cur.eigenvals()
    evs = []
    for val, mult in eigenvals_dict.items():
        for _ in range(mult):
            evs.append(float(val.evalf()))
            
    print('\nPoint: ', point, '\nhessian has eigenvals: ', evs)

    # Valuta il valore della funzione in quel punto
    f_val = float(f_sym.subs({x_sym: point[0], y_sym: point[1]}).evalf())
    print('The function has value: ', f_val)

    # Classifica il punto basato sui segni degli autovalori
    if np.sign(evs[0]) == np.sign(evs[1]):
        if np.sign(evs[0]) > 0:
            print('The point is a MINIMA')
            points_type.append('MINIMA')
        else:
            print('The point is a MAXIMA')
            points_type.append('MAXIMA')
    else:
        print('The point is a SADDLE POINT')
        points_type.append('SADDLE POINT')

# Plot "Contout plot" use "countour function"
import matplotlib.pyplot as plt
import numpy as np
# Create a grid of points
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
# Define the function
Z = 2*X**2 - 1.05*X**4 + X**6/6 + X*Y + Y**2
# Create the contour plot
plt.close('all')
plt.figure(figsize=(8, 6))
contour = plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
# Plot stationary points
for i, point in enumerate(stationary_points):
    x_pt = float(point[0].evalf())
    y_pt = float(point[1].evalf())
    if points_type[i] == 'MINIMA':
        plt.plot(x_pt, y_pt, 'ro', label='Minima' if i == 0 else "")
    elif points_type[i] == 'MAXIMA':
        plt.plot(x_pt, y_pt, 'bo', label='Maxima' if i == 0 else "")
    else:
        plt.plot(x_pt, y_pt, 'go', label='Saddle Point' if i == 0 else "")
plt.title('Contour Plot with Stationary Points')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plot surface plot using matplotlib
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title('Surface Plot')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
# Plot stationary points on surface
for i, point in enumerate(stationary_points):
    x_pt = float(point[0].evalf())
    y_pt = float(point[1].evalf())
    z_pt = float(f_sym.subs({x_sym: point[0], y_sym: point[1]}).evalf())
    if points_type[i] == 'MINIMA':
        ax.scatter(x_pt, y_pt, z_pt, color='r', s=100, label='Minima' if i == 0 else "")
    elif points_type[i] == 'MAXIMA':
        ax.scatter(x_pt, y_pt, z_pt, color='b', s=100, label='Maxima' if i == 0 else "")
    else:
        ax.scatter(x_pt, y_pt, z_pt, color='g', s=100, label='Saddle Point' if i == 0 else "") 
plt.show()


# ! Newton method to find the minima:
def Newton(f_sim, gradient, hessian, x0, tol, max_iter):
    x = x0
    x_history = list()
    j_history = list()
    x_history.append(x)
    j_history.append(float(f_sim(x)))
    for i in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        incr = np.linalg.solve(hess, -grad)
        x = x + incr
        x_history.append(x)
        j_history.append(float(f_sim(x)))
        if np.linalg.norm(grad) < tol:
            print(f'Converged in {i} iterations.')
            return x_history, j_history
    print('Maximum iterations reached.')
    return x_history, j_history

tol = 1e-6
max_iter = 1000
x0 = [(2.5, -2.5), (0.8, -2.5), (-0.4, -2.5)]

# Definisco le funzioni JAX fuori dal ciclo
f_sim_jax = lambda v: 2*v[0]**2 - 1.05*v[0]**4 + v[0]**6/6 + v[0]*v[1] + v[1]**2
f_sim_jit = jax.jit(f_sim_jax)
gradient_jit = jax.jit(jax.grad(f_sim_jax))
hessian_jit = jax.jit(jax.hessian(f_sim_jax))

plt.close('all')
for x_start in x0:
    x_start = np.array(x_start)
    x_history, j_history = Newton(f_sim_jit, gradient_jit, hessian_jit, x_start, tol, max_iter)

    # Plot convergence results in a single coherent figure
    plt.figure(figsize=(18, 5))
    x_history = np.array(x_history)
    
    # 1. 2D Path on Contour Plot
    plt.subplot(1, 3, 1)
    plt.contour(X, Y, Z, levels=50, cmap='viridis', alpha=0.4)
    plt.plot(x_history[:, 0], x_history[:, 1], 'r-', linewidth=1, label='Path')
    
    # Add stationary points to 2D
    for i, point in enumerate(stationary_points):
        x_pt, y_pt = float(point[0].evalf()), float(point[1].evalf())
        color = 'r' if points_type[i] == 'MINIMA' else 'b' if points_type[i] == 'MAXIMA' else 'g'
        plt.plot(x_pt, y_pt, color + 'o', markersize=5)

    plt.plot(x_history[0, 0], x_history[0, 1], 'kx', markersize=10, label='Start')
    plt.title(f'Convergence Path from {x_start}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    # 2. Function Value History
    plt.subplot(1, 3, 2)
    plt.semilogy(j_history, 'o-')
    plt.title('Value History (log scale)')
    plt.xlabel('Iteration')
    plt.ylabel('f(x,y)')

    # 3. 3D Surface with Path
    ax = plt.subplot(1, 3, 3, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, rstride=20, cstride=20)
    ax.plot(x_history[:, 0], x_history[:, 1], j_history, 'r.-', linewidth=2, label='Path')
    
    # Add stationary points to 3D
    for i, point in enumerate(stationary_points):
        x_pt = float(point[0].evalf())
        y_pt = float(point[1].evalf())
        z_pt = float(f_sym.subs({x_sym: point[0], y_sym: point[1]}).evalf())
        color = 'r' if points_type[i] == 'MINIMA' else 'b' if points_type[i] == 'MAXIMA' else 'g'
        ax.scatter(x_pt, y_pt, z_pt, color=color, s=50, label=points_type[i] if x_start is x0[0] else "")
    
    ax.set_title(f'3D View from {x_start}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.show()

# Note only the last point converge to the global minima, the other one converge to local minima or saddle point