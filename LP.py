# This code implements some interior point methods from the book
# Primal-Dual Interior Point Methods by Wright.

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def form_KKT_matrix(A, x, s):
    m, n = A.shape
    # Should these zero matrices be constructed one time only outside of this function?
    return np.block([[np.zeros((n, n)), A.T, np.eye(n)],
                  [A, np.zeros((m, m)), np.zeros((m, n))],
                  [np.diag(s), np.zeros((n, m)), np.diag(x)]])
   

def solveLP_SPF(c, A, b, x, lmbda, s):
    max_iter = 500
    m, n = A.shape

    theta = .4
    sigma = 1 - .4 / np.sqrt(n)

    all_ones = np.ones(n)
    rhs = np.zeros(n + m + n)
    costs = []
    for k in range(max_iter):
        # M = np.block([[zeros_n, A.T, I_n], [A, zeros_m, zeros_mn], [S, zeros_nm, X]])
        M = form_KKT_matrix(A, x, s)

        mu = np.vdot(x, s) / n
        rhs[m+n:] = -x * s + sigma * mu * all_ones
        sln = np.linalg.solve(M, rhs)
        dx = sln[0:n]
        dlmbda = sln[n:n+m]
        ds = sln[n+m:]

        x = x + dx
        lmbda = lmbda + dlmbda
        s = s + ds

        costs.append(np.vdot(c, x))

    return x, costs

def solveLP_MPC(c, A, b, x, lmbda, s):
    # We minimize c^T x subject to Ax = b, x >= 0 using Algorithm MPC.
    max_iter = 50
    m, n = A.shape
    
n = 10
m = 5

A = np.random.randn(m, n)
lmbda = np.random.randn(m)
x = np.random.rand(n)
s = np.random.rand(n)
b = A @ x
c = A.T @ lmbda + s

x_SPF, costs = solveLP_SPF(c, A, b, x, lmbda, s)
print(f'minimum value from SPF: {np.vdot(x_SPF, c)}')
plt.figure()
plt.plot(costs)
plt.show()

M_check = form_KKT_matrix(A, x, s)

x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x == b, x >= 0])
prob.solve(tol_feas=1e-9, tol_gap_rel=1e-9)
print(f'Minimum value from CVXPY: {prob.value}')
x_CVX = x.value

error = np.linalg.norm(x_SPF - x_CVX) / np.linalg.norm(x_CVX)
print(f'Relative error is: {error}')
















