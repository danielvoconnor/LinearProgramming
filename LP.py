# This code implements some interior point methods from the book
# Primal-Dual Interior Point Methods by Wright.

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import timeit

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

def find_alpha(x, dx):
    alpha = np.inf
    for i in range(len(dx)):
        if dx[i] < 0:
            alpha = min(alpha, x[i] / (-dx[i]))

    return alpha

def solveLP_MPC(c, A, b, x, lmbda, s):
    # We minimize c^T x subject to Ax = b, x >= 0 using Algorithm MPC.
    max_iter = 20
    m, n = A.shape
    xk = x.copy()
    lmbdak = lmbda.copy()
    sk = s.copy()
    mu_vals = []
    
    for k in range(max_iter):
        rb = A @ xk - b
        rc = A.T @ lmbdak + sk - c
        M = form_KKT_matrix(A, xk, sk)
        rhs = np.concat((-rc, -rb, -xk * sk))
        sln = np.linalg.solve(M, rhs)
        dx_aff = sln[0:n]
        dlmbda_aff = sln[n:n+m]
        ds_aff = sln[n+m:]
        alpha_aff_primal = find_alpha(xk, dx_aff)
        alpha_aff_dual = find_alpha(sk, ds_aff)
        mu_aff = np.vdot(xk + alpha_aff_primal * dx_aff, sk + alpha_aff_dual * ds_aff) / n
        mu = np.vdot(xk, sk) / n
        sigma = (mu_aff / mu)**3
        
        rhs = np.concat((np.zeros(n), np.zeros(m), -dx_aff*ds_aff + sigma*mu))
        sln = np.linalg.solve(M, rhs)
        dx_cc = sln[0:n]
        dlmbda_cc = sln[n:n+m]
        ds_cc = sln[n+m:]
        
        dxk = dx_aff + dx_cc
        dlmbdak = dlmbda_aff + dlmbda_cc
        dsk = ds_aff + ds_cc

        alpha_max_primal = find_alpha(xk, dxk)
        alpha_max_dual = find_alpha(sk, dsk)
        alphak_primal = min(.99 * alpha_max_primal, 1.0)
        alphak_dual = min(.99 * alpha_max_dual, 1.0)
        xk = xk + alphak_primal * dxk
        lmbdak = lmbdak + alphak_dual * dlmbdak
        sk = sk + alphak_dual * dsk

        mu_vals.append(mu)

    return xk, mu_vals

        
        
    
n = 500
m = 300

A = np.random.randn(m, n)
lmbda = np.random.randn(m)
x = np.random.rand(n)
s = np.random.rand(n)
b = A @ x
c = A.T @ lmbda + s

#x_SPF, costs = solveLP_SPF(c, A, b, x, lmbda, s)
#print(f'minimum value from SPF: {np.vdot(x_SPF, c)}')
#plt.figure()
#plt.plot(costs)
#plt.show()

time_start = timeit.default_timer()
x_MPC, mu_vals = solveLP_MPC(c, A, b, x, lmbda, s)
time_MPC = timeit.default_timer() - time_start
#plt.figure()
#plt.plot(mu_vals)
#plt.show()

rb = np.linalg.norm(A @ x_MPC - b)
cost_MPC = np.vdot(c, x_MPC)
min_x_MPC = np.min(x_MPC)
print(f'rb (MPC) is: {rb}')
print(f'Is x_MPC nonnegative? {min_x_MPC}')
print(f'Minimum value from MPC is: {cost_MPC}')


time_start = timeit.default_timer()
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x == b, x >= 0])
prob.solve(tol_feas=1e-9, tol_gap_rel=1e-9)
time_CVX = timeit.default_timer() - time_start
print(f'Minimum value from CVXPY: {prob.value}')
x_CVX = x.value
print(f'time_MPC is: {time_MPC}')
print(f'time_CVX is: {time_CVX}')
print(f'ratio is: {time_CVX / time_MPC}')

# error_SPF = np.linalg.norm(x_SPF - x_CVX) / np.linalg.norm(x_CVX)
error_MPC = np.linalg.norm(x_MPC - x_CVX) / np.linalg.norm(x_CVX)
# print(f'Relative error (SPF) is: {error_SPF}')
print(f'Relative error (MPC) is: {error_MPC}')
















