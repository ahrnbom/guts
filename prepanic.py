""" 
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    Matlab's ccode function does not actually produce C code, but rather some 
    C-like pseudocode which we will denote MC code. This module converts MC code
    into actual C code that can be compiled. Other than the output from ccode, 
    the MC code should contain two lines specifying the name of inputs, 
    and wether the output (which must be a single variable!) is a scalar 
    (denoted t0) or a 2D matrix (denoted A0)


    Note: 
    MATLAB assumes that non-set values in a matrix should be zero (which is 
    really stupid in C or C-like code) so you have to always initialize the 
    output with np.zeros, never with np.empty!
"""

from pathlib import Path

def present_inputs(inputs):
    cds = [f"const double {inp}" for inp in inputs]
    return ', '.join(cds)

def present_data(inputs, start_at=0):
    l = len(inputs)
    d = [f"data[{i+start_at}]" for i in range(l)]
    return ', '.join(d)

def precompile(mc_path, verbose=False):
    if verbose:
        print(f"Compiling MC code in {mc_path} into C code...")
    
    assert(mc_path.is_file())
    lines = [line for line in mc_path.read_text().split('\n') if line]
    
    assert(len(lines) > 2)
    inputs_line = lines[0]
    output_line = lines[1]
    
    is_single_output = output_line.split(': ')[-1]=='0'
    if not is_single_output:
        output_size = [int(x) for x in output_line.split(': ')[-1].split() if x]
    
    inputs = [name.strip() for name in inputs_line.split(': ')[1].split(',')]
    
    lines = lines[2:]
    
    out_lines = ['#include <math.h>', '']
    if is_single_output:
        out_lines.append(f"double generated_function({present_inputs(inputs)}) " + '{')
    else:
        out_lines.append(f"void generated_function(double* A0, {present_inputs(inputs)}) " + '{')
    
    width_so_far = 0
    height_so_far = 0
    for line in lines:
        lhs, rhs = line.split(' = ')
        if lhs.startswith('A0'):
            assert(not is_single_output)
            l_split = [x for x in lhs.replace(']', '[').split('[') if x]
            y = int(l_split[1])
            x = int(l_split[2])
            
            if (x+1) > width_so_far:
                width_so_far = x+1
            if (y+1) > height_so_far:
                height_so_far = y+1
                
            pos = y*width_so_far + x
            
            out_lines.append(f"    A0[{pos}] = {rhs}")
        else:
            assert(lhs.startswith('t'))
            out_lines.append(f"    const double {line}")
    
    if is_single_output:
        out_lines.append("    return t0;")
    
    out_lines.append("}")
    out_lines.append("")
    
    if is_single_output:
        out_lines.append("void main(double* data) {")
        out_lines.append(f"    double out = generated_function({present_data(inputs, start_at=1)});")
        out_lines.append("    data[0] = out;")
        out_lines.append("}")
    else:
        out_lines.append("void main(double* out, double* data) {")
        out_lines.append(f"    generated_function(out, {present_data(inputs)});")
        out_lines.append("}")
        out_lines.append(f"/* Output matrix should be of size {output_size[0]} x {output_size[1]} */")
        assert(output_size[0] >= height_so_far)
        assert(output_size[1] >= width_so_far)
    
    out_path = mc_path.with_suffix(".c")
    out_path.write_text("\n".join(out_lines))
    
    if verbose:
        print(f"It is written: {out_path}")
            
        
if __name__=="__main__":
    precompile(Path('generated') / 'gtriangul1_res.mc', verbose=True)
    