# This code implements some interior point methods from the book
# Primal-Dual Interior Point Methods by Wright.

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import timeit

def form_KKT_matrix(A, x, s):
    m, n = A.shape
    # Should these zero matrices be constructed one time only outside of this function?
    return np.block([[np.zeros((n, n)), A.T, np.eye(n)],
                  [A, np.zeros((m, m)), np.zeros((m, n))],
                  [np.diag(s), np.zeros((n, m)), np.diag(x)]])
   
def form_KKT_matrix_for_QP(Q, A, x, s):
    m, n = A.shape
    # Should these zero matrices be constructed one time only outside of this function?
    return np.block([[-Q, A.T, np.eye(n)],
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

def solve_KKT_system(L, low, x, s, rb, rc, rxs):
    # See equations 11.3a-c (p. 210) in Primal-Dual Interior Point Methods by Wright.
    rhs = -rb + A @ (-(x / s) * rc + rxs / s)
    dlmbda = sp.linalg.cho_solve((L, low), rhs)
    ds = -rc - A.T @ dlmbda
    dx = -(rxs + x * ds) / s
    return dx, dlmbda, ds

#def solve_KKT_system_for_QP(L, low, x, s, u, v, w):
    

def solveLP_MPC(c, A, b, x, lmbda, s):
    # We minimize c^T x subject to Ax = b, x >= 0 using Algorithm MPC
    # from chapter 10 (p. 198) of Primal-Dual Interior Point Methods by Wright.
    max_iter = 20
    tol = 1e-9
    m, n = A.shape
    xk = x.copy()
    lmbdak = lmbda.copy()
    sk = s.copy()
    mu_vals = []
    
    for k in range(max_iter):
        rb = A @ xk - b
        rc = A.T @ lmbdak + sk - c
 
        # M = form_KKT_matrix(A, xk, sk)
        #rhs = np.concat((-rc, -rb, -xk * sk))
        #sln = np.linalg.solve(M, rhs)
        #dx_aff = sln[0:n]
        #dlmbda_aff = sln[n:n+m]
        #ds_aff = sln[n+m:]

        # First we solve equation (10.1) in Wright, using the technique described in chapter 11.
        M = (A * (xk / sk)) @ A.T # See equation 11.3a (p. 210) in 
                                  # Primal-Dual Interior Point Methods by Wright.
        L, low = sp.linalg.cho_factor(M) # low is "true" if L is lower triangular.
        dx_aff, dlmbda_aff, ds_aff = solve_KKT_system(L, low, xk, sk, rb, rc, xk*sk)

        alpha_aff_primal = find_alpha(xk, dx_aff)
        alpha_aff_dual = find_alpha(sk, ds_aff)
        mu_aff = np.vdot(xk + alpha_aff_primal * dx_aff, sk + alpha_aff_dual * ds_aff) / n
        mu = np.vdot(xk, sk) / n
        sigma = (mu_aff / mu)**3
        
        #rhs = np.concat((np.zeros(n), np.zeros(m), -dx_aff*ds_aff + sigma*mu))
        #sln = np.linalg.solve(M, rhs)
        #dx_cc = sln[0:n]
        #dlmbda_cc = sln[n:n+m]
        #ds_cc = sln[n+m:]

        # Now we solve equation (10.7) in Wright, again using the chapter 11 technique.
        dx_cc, dlmbda_cc, ds_cc = \
            solve_KKT_system(L, low, xk, sk, np.zeros(m), np.zeros(n), dx_aff * ds_aff - sigma * mu) 
        
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
        if abs(mu) < tol: # mu should be positive, 
                          #but what if roundoff error makes it negative, is that possible?
             break 

    return xk, mu_vals

def solveQP_MPC(Q, c, A, b, x, lmbda, s):
    # We minimize (1/2)x^T Q x + c^T x subject to Ax = b, x >= 0 using Algorithm MPC
    # from chapter 10 (p. 198) of Primal-Dual Interior Point Methods by Wright.
    # Q is a symmetric positive semidefinite matrix.

    max_iter = 20
    tol = 1e-9
    m, n = A.shape
    xk = x.copy()
    lmbdak = lmbda.copy()
    sk = s.copy()
    mu_vals = []
    
    for k in range(max_iter):
        rb = A @ xk - b
        rc = -Q @ xk + A.T @ lmbdak + sk - c

        # First we solve equation (10.1) in Wright, using the technique described in chapter 11.
        # See p. 210 in Wright.
        # A slower but relatively clear way to solve the KKT system is commented out below.
#        M = form_KKT_matrix_for_QP(Q, A, xk, sk)
#        rhs = np.concat((-rc, -rb, -xk * sk))
#        sln = np.linalg.solve(M, rhs)
#        dx_aff = sln[0:n]
#        dlmbda_aff = sln[n:n+m]
#        ds_aff = sln[n+m:]

        # The code below could probably be cleaned up or clarified a bit.
        # We should not form the inverse of H explicitly.
        # Should I have a function that solves the KKT system with a given right hand side? Perhaps.
        H = Q + np.diag(sk / xk)
        Hinv = np.linalg.inv(H)
        AHinv = A @ Hinv
        mtrx = AHinv @ A.T
        L, low = sp.linalg.cho_factor(mtrx)
        rhs = -rb + AHinv @ (-rc + sk)
        dlmbda_aff = sp.linalg.cho_solve((L, low), rhs)
        dx_aff = Hinv @ (A.T @ dlmbda_aff - sk + rc)
        ds_aff = -sk - (sk / xk) * dx_aff
        # You can check that dlmbda_aff, dx_aff, and ds_aff here agree with the values
        # obtained using the slower but more clear method which is commented out above.
         
        alpha_aff_primal = find_alpha(xk, dx_aff)
        alpha_aff_dual = find_alpha(sk, ds_aff)
        mu_aff = np.vdot(xk + alpha_aff_primal * dx_aff, sk + alpha_aff_dual * ds_aff) / n
        mu = np.vdot(xk, sk) / n
        sigma = (mu_aff / mu)**3
        
        #######
        term  = (sigma * mu - dx_aff * ds_aff) / xk
        rhs = AHinv @ ( -term)
        dlmbda_cc = sp.linalg.cho_solve((L, low), rhs)
        dx_cc = Hinv @ (A.T @ dlmbda_cc + term)
        ds_cc = term - (sk / xk) * dx_cc
        #######
        # The slower but relatively clear way to solve the KKT system is commented out below.
        # You can check that dlmbda_cc, dx_cc, and ds_cc computed above agree with the
        # values obtained by the slower but more clear method below.
#        rhs = np.concat((np.zeros(n), np.zeros(m), -dx_aff*ds_aff + sigma*mu))
#        sln = np.linalg.solve(M, rhs)
#        dx_cc = sln[0:n]
#        dlmbda_cc = sln[n:n+m]
#        ds_cc = sln[n+m:]

        # Now we solve equation (10.7) in Wright, again using the chapter 11 technique.
#        dx_cc, dlmbda_cc, ds_cc = \
#            solve_KKT_system(L, low, xk, sk, np.zeros(m), np.zeros(n), dx_aff * ds_aff - sigma * mu) 
        
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
        if abs(mu) < tol: # mu should be positive, 
                          #but what if roundoff error makes it negative, is that possible?
             break 

    return xk, mu_vals




test_LP = False
test_QP = True

        
n = 500
m = 200

if test_QP:

    Q = np.random.randn(n, n)
    Q = Q.T @ Q
    A = np.random.randn(m, n)
    b = A @ np.random.rand(n) # This way of creating b guarantees the LP is feasible.
    c = np.random.rand(n) # Choosing c >= 0 guarantees the LP has a solution.
    
    #x_SPF, costs = solveLP_SPF(c, A, b, x, lmbda, s)
    
    x0 = np.random.rand(n)
    s0 = np.random.rand(n)
    lmbda0 = np.random.randn(m)
    time_start = timeit.default_timer()
    # x_MPC, mu_vals = solveLP_MPC(c, A, b, x0, lmbda0, s0)
    x_MPC, mu_vals = solveQP_MPC(Q, c, A, b, x0, lmbda0, s0) 
    time_MPC = timeit.default_timer() - time_start
    #plt.figure()
    #plt.plot(mu_vals)
    #plt.show()
    
    rb = np.linalg.norm(A @ x_MPC - b)
    cost_MPC = .5 * np.vdot(x_MPC, Q @ x_MPC) + np.vdot(c, x_MPC)
    min_x_MPC = np.min(x_MPC)
    print(f'rb (MPC) is: {rb}')
    print(f'Is x_MPC nonnegative? {min_x_MPC}')
    print(f'Minimum value from MPC is: {cost_MPC}')
    
    Q = cp.psd_wrap(Q) 
    time_start = timeit.default_timer()
    x = cp.Variable(n)
    # prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x == b, x >= 0])
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + c.T @ x), [A @ x == b, x >= 0])
    
    # prob.solve(tol_feas=1e-9, tol_gap_rel=1e-9)
    prob.solve()
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
    
    

##############################
if test_LP:
    A = np.random.randn(m, n)
    b = A @ np.random.rand(n) # This way of creating b guarantees the LP is feasible.
    c = np.random.rand(n) # Choosing c >= 0 guarantees the LP has a solution.
    
    #x_SPF, costs = solveLP_SPF(c, A, b, x, lmbda, s)
    
    x0 = np.random.rand(n)
    s0 = np.random.rand(n)
    lmbda0 = np.random.randn(m)
    time_start = timeit.default_timer()
    x_MPC, mu_vals = solveLP_MPC(c, A, b, x0, lmbda0, s0)
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
    
    # prob.solve(tol_feas=1e-9, tol_gap_rel=1e-9)
    prob.solve()
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
    










