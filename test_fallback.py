"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Tests of the fallback assignment code that runs when scipy's hungarian
    step fails.
"""

import numpy as np

from hungarian import fallback_assignment

def same_elements(A, B):
    if not (len(A) == len(B)):
        return False
    
    for a, b in zip(A, B):
        if not (a==b):
            return False
    
    return True 

def all_unique(A):
    n = len(A)
    m = len(list(set(A)))
    return n==m

def test_fallback():
    A = np.array([[1,2,3], [3,2,1], [100,1,100]], dtype=np.float32)
    a1, a2 = fallback_assignment(A)
    
    assert(all_unique(a1))
    assert(all_unique(a2))
    
    assert same_elements(a1, [0, 1, 2])
    assert same_elements(a2, [0, 2, 1])

    B = np.array([[np.nan, np.nan, 0.1],
                  [0.1, np.nan, np.nan],
                  [np.nan, np.inf, np.nan],
                  [np.nan, 0.3, np.nan]], dtype=np.float32)
    b1, b2 = fallback_assignment(B)
    
    assert(all_unique(b1))
    assert(all_unique(b2))
    
    assert same_elements(b1, [0, 1, 3])
    assert same_elements(b2, [2, 0, 1])


if __name__=="__main__":
    test_fallback()