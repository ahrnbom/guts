"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Utility functions used all over the codebase
"""


from typing import List
import numpy as np
import os

# Call this to make sure a long running script doesn't slow down other things running on the same computer
def nice():
    os.nice(10)

# Builds one big list containing all elements of several lists
def all_elements(lists:List[List]):
    new_list = list()
    for l in lists:
        new_list.extend(l)
    return new_list

# Divides a list into sublists of (max) length n 
# The last sublist will contain any leftovers, and may be shorter than n 
def list_batches(some_list:List, n:int):
    N = len(some_list)
    out = list()
    for i in range(N//n):
        sublist = some_list[i*n:(i+1)*n]
        out.append(sublist)
    
    if len(out)*n < N:
        # Take care of leftovers
        sublist = some_list[len(out)*n:]
        out.append(sublist)
        
    return out 

# Go through an iterator data and divide it into lists of length (at most) bs 
def batches(data, bs):
    out = list()
    for d in data:
        out.append(d)
        if len(out) == bs:
            yield out 
            out = list() 
    if out:
        yield out 

def unique_elements(data:list):
    return list(set(data))

# Removes the intro from text, if it exists at the beginning, otherwise raises error
# Essentially, it works like you'd expect the built-in lstrip to work
def good_lstrip(text, intro):
    assert(len(intro) <= len(text))
    l = len(intro)
    first = text[:l]
    assert(first == intro)

    return text[l:]

def vector_normalize(x):
    n = np.linalg.norm(x)
    if n > 0.0000001:
        return x/n
    else:
        x = np.zeros_like(x)
        x[0] = 1
        return x 

def vector_dist(a, b):
    a = np.squeeze(a)
    b = np.squeeze(b)
    return np.linalg.norm(a-b)

# Matlab's A\B
def ldiv(A, B):
    # A\B = x
    # B = A*x
    return np.linalg.lstsq(A, B, rcond=None)[0]

# Matlab's A/B
def rdiv(A,B):
    # A/B = x
    # A = x*B = B'*x'
    return np.linalg.lstsq(B.T, A.T, rcond=None)[0].T

# Finds vector v such that X @ v is minimized
def dlt(X):
    _,_,vh = np.linalg.svd(X, full_matrices=True) 
    return vh[-1, :]

def append_ones(X):
    if len(X.shape) == 2:
        n,m = X.shape
        return np.vstack([X, np.ones((m,), dtype=X.dtype)])
    elif len(X.shape) == 1:
        n = len(X)
        Y = np.ones((n+1, 1), dtype=X.dtype)
        Y[:n, 0] = X
        return Y

def long_str(num:int, N:int=6):
    s = str(num)
    
    n = N - len(s)
    if n > 0:
        s = '0'*n + s
    
    return s

def clamp(x, xmin, xmax):
    l = [xmin, x, xmax]
    l.sort()
    return l[1]
    
def intr(x):
    return int(round(x))

def pflat(x):
    if len(x.shape) == 1:
        x /= x[-1]
    else:
        x /= x[-1, :]
    
    return x

# Finds the tightest AABB around a segmentation mask 
def smallest_box(a):
    r = a.any(1)
    if r.any():
        m,n = a.shape
        c = a.any(0)
        y1 = r.argmax()
        y2 = m-r[::-1].argmax()
        x1 = c.argmax()
        x2 = n-c[::-1].argmax()
        out = np.array([x1, y1, x2, y2])
    else:
        out = None
    return out

def box_contains(box, point):
    x1, y1, x2, y2 = box 
    x = point[0]
    y = point[1]

    if (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2):
        return True 
    
    return False 
    
def mask_centroid(mask):
    x1, y1, x2, y2 = smallest_box(mask)
    return (x1 + x2)/2, (y1 + y2)/2

def aabb_centroid(box):
    x1, y1, x2, y2 = box 
    return (x1 + x2)/2, (y1 + y2)/2

def natural_choice(some_list):
    return some_list[len(some_list)//2]

# Takes many numpy arrays and flattens them all into a long horizontal vector
def multi_flatten(Xs):
    out = []
    for X in Xs:
        if isinstance(X, (int, float)):
            out.append(X)
        else:
            s = X.shape    
            if len(s) == 2:
                n,m = s
                for y in range(n):
                    for x in range(m):
                        out.append(X[y,x])
            elif len(s) == 1:
                for i in range(s[0]):
                    out.append(X[i])
            else:
                raise ValueError("Multi_flatten only works for 1D and 2D matrices and single values")
    return np.array(out, dtype=np.float64)

def rotation_matrix(angle:float):
    s = np.sin(angle)
    c = np.cos(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

def cv_point(x, y):
    return (intr(float(x)), intr(float(y)))

# Converts midpoint and width/height to AABB (x1, y1, x2, y2)
def to_aabb(x, y, w, h):
    w2 = w/2.0
    h2 = h/2.0
    x1 = x - w2
    y1 = y - h2 
    x2 = x + w2 
    y2 = y + h2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def dict_copy(some_dict:dict):
    new_dict = dict()
    for key, val in some_dict.items():
        new_dict[key] = val
    
    return new_dict

def dict_merge(a:dict, b:dict):
    # Move any entries from b into a 
    for key, val in b.items():
        if not key in a:
            a[key] = val
    
    return a 

def angle_distance(a, b):
    # Distance between two angles, [-pi, pi)
    return np.arctan2(np.sin(a-b), np.cos(a-b))

# Essentially factor*a + (1-factor)*b but for angles in radians 
def weighted_angle(a, b, factor):
    if factor > 0.999:
        return a
    elif factor < 0.001:
        return b 

    A = np.array([np.cos(a), np.sin(a)], dtype=np.float32)
    B = np.array([np.cos(b), np.sin(b)], dtype=np.float32)

    C = factor * A + (1.0-factor) * B 
    if np.linalg.norm(C) > 0.001:
        c = np.arctan2(C[1], C[0])
        return c 
    else:
        # Very rough approximation
        if factor > 0.5:
            return a 
        return b 

# Reasonable number of threads for multiprocessing pools
def thread_count():
    # The idea is that you might want to leave a core or two for other 
    # apps to keep the computer responsive
    c = os.cpu_count()
    if c > 4:
        return c-2
    elif c == 4:
        return 3
    else:
        return c