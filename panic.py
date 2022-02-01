""" 
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Python API for Numpy arrays In C 
    Only works with Numpy arrays! The C function cannot take any other inputs
"""

import numpy as np
import ctypes
import subprocess
from pathlib import Path

from prepanic import precompile

def age(some_file):
    return some_file.stat().st_mtime

def do_compile(c_file, so_file, options=['-O2', '-fPIC', '-fopenmp'], deps=None, shutup=False):
    c_path = Path(c_file)
    if c_path.suffix == ".mc":
        precompile(c_path)
        c_path = c_path.with_suffix('.c')
        assert(c_path.is_file())
        c_file = str(c_path)
        
    cmd = ['gcc', '-shared', *options, '-o', so_file, c_file]
    
    if deps is not None:
        cmd.extend([f"{x.resolve()}" for x in deps])
        
    cmd = [str(x) for x in cmd if x]
    
    if not shutup:
        print(f"Compiling {c_file}, with command '{' '.join(cmd)}'...")
    ret = subprocess.run(cmd)
    if not ret.returncode == 0:
        raise ValueError(f"The C code file {c_file} does not compile, see errors above")
        
    if not so_file.is_file():
        raise ValueError(f"For some reason, {so_file} does not exist!")
    
    if not shutup:
        print("Compilation successful.")



def run(c_file, fun_name, args, dependencies=[]):
    c_file = Path(c_file)
    so_file = c_file.with_suffix('.so')
    
    c_files = ([Path(x) for x in dependencies])
    so_files = [x.with_suffix('.so') for x in c_files]
    for c, so in zip(c_files, so_files):
        if (not so.is_file()) or (age(so) < age(c)):
            do_compile(c, so)
    
    if (not so_file.is_file()) or (age(so_file) < age(c_file)):
        do_compile(c_file, so_file, deps=so_files)
    
    # Note that this makes a copy!! Without this, data cannot be accessed in C
    # for some images (but not for others). Not sure why, but maybe the OWNDATA flag is relevant for this mystery
    args = [np.ascontiguousarray(arg, dtype=arg.dtype).copy() for arg in args]
    
    # Need to add support for more dtypes? Add them here! You need to know the corresponding ctypes type
    type_to_c = {np.dtype('uint8'):   ctypes.c_ubyte,
                 np.dtype(bool):      ctypes.c_bool,
                 np.dtype('int32'):   ctypes.c_int,
                 np.dtype('uint32'):  ctypes.c_uint,
                 np.dtype('int64'):   ctypes.c_long,
                 np.dtype('uint64'):  ctypes.c_ulong,
                 np.dtype('float32'): ctypes.c_float,
                 np.dtype('float64'): ctypes.c_double}    
    
    # You could also do arg.ctypes.data but that doesn't work sometimes (??)    
    cargs = []
    for arg in args:
        if not arg.dtype in type_to_c:
            raise ValueError(f"Cannot send numpy array with dtype {arg.dtype} to C. Supported dtypes are: {list(type_to_c.keys())}")
        c_type = type_to_c[arg.dtype]
        pointer_type = ctypes.POINTER(c_type)
        c_data = arg.ctypes.data_as(pointer_type)
        cargs.append(c_data)
    
    lib = ctypes.cdll.LoadLibrary(so_file.resolve())
    fun = lib.__getattr__(fun_name)
    
    fun(*cargs)
    
    return args
