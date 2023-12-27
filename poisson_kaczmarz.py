from scipy.io import mmread
import numpy as np
from time import time
from numpy.linalg import norm
from scipy.sparse import csr_matrix
import scipy.linalg as sla

# Reading the files
A = mmread('poisson2D.mtx')
f = mmread('poisson2D_b.mtx')
A = csr_matrix(A)  

# Block Kaczmarz
block_numbers = [2, 4, 6, 8, 16]
kaczmarz_times = []
kaczmarz_residuals = []

# Pseudo-inverse solution for compare results
pinv_solution = sla.pinv(A.todense()) @ f
predicted_value = A @ pinv_solution
residual_pinv = norm(f - predicted_value)

for K in block_numbers:
    start_time = time()

    x_kaczmarz = np.zeros_like(f) # Initiate x as zero matrix
    max_iteration = 100 # Iteration number
    iteration = 0
    sum_delta = 0

    # Initiate delta value 
    delta = 0

    # Split A, f and x matrixes according to K
    num_rows = A.shape[0]
    split_size = num_rows // K
    splitted_matrix_A = []
    splitted_matrix_f = []

    for i in range(K):

        # Split A
        start_index_A = i * split_size
        end_index_A = (i + 1) * split_size if i < K - 1 else num_rows
        part_A = A[start_index_A:end_index_A, :]
        splitted_matrix_A.append(part_A)

        # Split f
        start_index_f = i * split_size
        end_index_f = (i + 1) * split_size if i < K - 1 else num_rows
        part_f = f[start_index_f:end_index_f, :]
        splitted_matrix_f.append(part_f)

    # Starts kaczmarz algorithm
    while iteration < max_iteration:
        for block_number in range(K):
            block = block_number % K

            # Get the parts
            block_matrix_A = splitted_matrix_A[block]
            block_matrix_f = splitted_matrix_f[block]

            # Apply the formula
            delta = delta + np.linalg.pinv(block_matrix_A.toarray()) @ (block_matrix_f - (block_matrix_A * delta))
            sum_delta += delta
                
        iteration = iteration + 1
      
        # Update solution matrix
        x_kaczmarz = x_kaczmarz + sum_delta

        # Compute the relative residual
        residual = norm(A @ x_kaczmarz - f) / norm(f)

        # If residual is small enough, than convergence succeed
        if residual < 1e-5:
           break
    
    elapsed_time = time() - start_time
    kaczmarz_times.append(elapsed_time)
    kaczmarz_residuals.append(residual)

    print(f"Block Kaczmarz (K={K})=> Time: {elapsed_time}, Residual: {residual}")

print("Scipy pseudo-inverse residual: ", residual_pinv)

# Fit a polynomial without K=6
coefficients_1 = np.polyfit(block_numbers[:3] + block_numbers[4:], kaczmarz_times[:3] + kaczmarz_times[4:], 2)
poly_1 = np.poly1d(coefficients_1)

# Estimate the time for K=6
estimated_time_k6 = poly_1(6)
print(f"Estimated time for K=6: {estimated_time_k6}")

# Fit a polynomial for all results
coefficients_2 = np.polyfit(block_numbers, kaczmarz_times, 2)
poly_2 = np.poly1d(coefficients_2)

# Extrapolate the time for K=16
estimated_time_k16 = poly_2(16)
print(f"Estimated time for K=16 (extrapolated): {estimated_time_k16}")






