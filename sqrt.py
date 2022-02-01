"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Implementations of the square root of matrices, used inside Kalman
    filters. Because scipy's cholesky isn't quite stable enough, this module's 
    implementation applies some hacks that ensure that an answer is essentially
    always provided, even if it may not be very good. The reasoning is that
    a bad estimation is probably better than a crash.
"""

import scipy
import numpy as np 

# Based on https://github.com/rlabbe/filterpy/issues/62
# and https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
# and https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
# To summarize, filterpy's implementation of UKF and Merwe's sigma points
# isn't quite numerically stable, but this hack fixes the problem
def sqrt_func(A):
    try:
        result = scipy.linalg.cholesky(A)
    except scipy.linalg.LinAlgError:
        # Big Brain

        B = (A + A.T) / 2
        
        # numpy's svd is faster but scipy's svd is more numerically stable
        _, s, V = scipy.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2

        if is_pos_def(A3):
            result = A3 
        else:
            spacing = np.spacing(np.linalg.norm(A))
            I = np.eye(A.shape[0])
            k = 1
            while not is_pos_def(A3):
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += I * (-mineig * k**2 + spacing)
                k += 1
            result = A3 

    return result 

def is_pos_def(B):
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
