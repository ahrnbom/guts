""" 
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.


    A module which provides an easy interface for matplotlib for simple plots 
    """

import matplotlib as mpl
# Allows Matplotlib to run without a GUI
mpl.use('Agg') 
import matplotlib.pyplot as plt

def multi_plot(xs, ys, filepath, xlabel=None, ylabel=None, linewidth=2, 
               xlim=None, ylim=None, style=None, styles=None, title=None, 
               legend=None, legend_outside=False, use_grid=False):

    fig = plt.figure()
    ax = fig.add_subplot(111) 
    
    if xs is None:
        xs = []
        for y in ys:
            xs.append( list(range(1,len(y)+1)) ) # We expect the first x-value to be 1, not 0.
    
    lines = []
    if styles is None:
        for x, y in zip(xs, ys):
            if style is None:
                # Who thought it was a good idea to return a tuple from ax.plot?
                # This syntax is error-prone
                curr_line, = ax.plot(x,y, linewidth=linewidth)
            else:
                curr_line, = ax.plot(x,y, style, linewidth=linewidth)
                
            lines.append(curr_line)
    else:
        for x, y, style in zip(xs, ys, styles):
            curr_line, = ax.plot(x, y, style, linewidth=linewidth)
            lines.append(curr_line)
            
    if not (xlabel is None):
        plt.xlabel(xlabel)
    
    if not (ylabel is None):
        plt.ylabel(ylabel)

    if not (xlim is None):
        plt.xlim(xlim)
    
    if not (ylim is None):
        plt.ylim(ylim)    
        
    if not (title is None):
        plt.title(title)
        
    if not (legend is None):
        if legend_outside:
            plt.legend(lines, legend, bbox_to_anchor=(1.0, 1.0))
        else:
            plt.legend(lines, legend)

    if use_grid:
        plt.grid(True)

    # dpi sets the size of the output image
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight') 
    plt.close(fig)

# Takes two lists, x and y, and draws a simple plot.
def simple_plot(x, y, filepath, xlabel=None, ylabel=None, linewidth=2, 
                xlim=None, ylim=None, style=None):
    multi_plot([x], [y], filepath, xlabel=xlabel, ylabel=ylabel, 
               linewidth=linewidth, xlim=xlim, ylim=ylim, style=style)

# Draws a simple histogram
def histogram_plot(x, filepath, n=64, title=""):
    fig = plt.figure()
    plt.hist(x, bins=n)
    if title:
        plt.title(title)
    plt.savefig(str(filepath), dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    simple_plot([1,2,3,4,5],[1,2,1,1,2], 'anteaters.png', 
                xlabel='number of ant-eaters', ylabel='number of ants')