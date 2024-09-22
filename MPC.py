# This code implements Algorithm MPC (Mehrotra's predictor-corrector algorithm)
# on p. 198 of the book Primal-Dual Interior Point Methods by Wright.

import numpy as np
import matplotlib.pyplot as plt

def MPC(c, A, b, x0, lmbda0, s0):
    # We minimize c^T x subject to Ax = b, x >= 0 using Algorithm MPC.
     

    
    
