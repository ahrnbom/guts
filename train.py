"""
    Copyright (C) 2022 Martin Ahrnbom
    Released under MIT License. See the file LICENSE for details.
    
    
    General script for training a CNN in PyTorch
"""

from typing import List
import torch 
from torch import optim
import numpy as np 
from datetime import datetime 

from plot import multi_plot
from util import batches 

    
def train(net, data, folder, criterion, write=print, 
          batch_size=16, learning_rate=1e-5, epochs=64,
          plot_title="Network Training"):

    write(f"Starting training at {datetime.now()}")
    write(f"Learning rate: {learning_rate}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    write(f"Using PyTorch device {device}")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    loss_history = list()
    val_history = list()

    max_n_train_batches = 0
    
    for epoch in range(epochs):
        epoch_start = datetime.now()

        # Train!
        net.train()
        loss_sum = 0.0
        n_train_batches = 0
        for batch_num, train_batch in enumerate(batches(data['training'](), 
                                                        batch_size)):
            batch_length = len(train_batch) # not always batch size!
            
            xs = [t[0] for t in train_batch]
            ys = [t[1] for t in train_batch]
            # xs and ys are now lists of tuples like (x1, x2...)

            # Build batches in numpy 
            nx = len(xs[0]) # 2 would mean we have x1, x2 as inputs to network
            xx = [np.stack([x[i] for x in xs]) for i in range(nx)]

            ny = len(ys[0]) # 2 would mean we have y1, y2 as ground truth(s)
            yy = [np.stack([y[i] for y in ys]) for i in range(ny)]

            # Now xx is a list of the inputs, of shape (bs, ...)
            # Same for y, except it's ground truth 

            # Convert to PyTorch
            xx = [torch.from_numpy(x).to(device) for x in xx]
            yy = [torch.from_numpy(y).to(device) for y in yy]

            optimizer.zero_grad()
            outputs = net(*xx)
            loss = criterion(outputs, *yy)
            loss.backward()
            optimizer.step()

            curr_loss = float(loss.detach().cpu()) / batch_length
            loss_sum += curr_loss
            mean_loss = loss_sum/(batch_num+1)
            n_train_batches += 1
            max_n_train_batches = max(max_n_train_batches, n_train_batches)

            if (batch_num%200 == 0):
                write(f"Epoch {epoch+1}, " \
                      f"Batch {batch_num+1} / {max_n_train_batches}, "\
                      f"Loss {mean_loss:_}")
                # max_n_train_batches will be wrong during first batch...

        # Validate!
        val_loss = 0.0
        net.eval()
        n_val_batches = 0
        for batch_num, val_batch in enumerate(batches(data['validation'](), 
                                                      batch_size)):
            batch_length = len(val_batch)
            xs = [t[0] for t in val_batch]
            ys = [t[1] for t in val_batch]
            
            nx = len(xs[0]) 
            xx = [np.stack([x[i] for x in xs]) for i in range(nx)]

            ny = len(ys[0]) 
            yy = [np.stack([y[i] for y in ys]) for i in range(ny)]

            xx = [torch.from_numpy(x).to(device) for x in xx]
            yy = [torch.from_numpy(y).to(device) for y in yy]

            outputs = net(*xx)
            loss = criterion(outputs, *yy)
            curr_loss = float(loss.detach().cpu()) / batch_length
            val_loss += curr_loss 
            n_val_batches += 1 
        
        val_loss /= n_val_batches
        train_loss = loss_sum / n_train_batches
        write(f"Epoch {epoch+1}/{epochs} done. Train loss: {train_loss:_}, " \
              f"val loss: {val_loss}")

        # Store and visualize
        loss_history.append(train_loss)
        val_history.append(val_loss)
        n_history = len(loss_history)
        if n_history > 2:
            multi_plot([range(1, n_history+1), range(1, n_history+1)], 
                       [loss_history, val_history], folder / "train_plot.png", 
                       xlabel='Epochs', ylabel='Loss', 
                       legend=['Training loss', 'Validation loss'], 
                       title=plot_title, use_grid=True,
                       ylim=[0.0, max(max(val_history), max(loss_history))*1.1])
                       
            write("Plot drawn!")
        
        w_path = folder / f"epoch{epoch+1}_vloss{val_loss:_}.pth"
        torch.save(net.state_dict(), w_path)
        write(f"It is written... {w_path}")

        write(f"Best val loss so far: {np.min(val_history)} at epoch " \
              f"{np.argmin(val_history)+1}")

        now = datetime.now()
        epoch_time = now - epoch_start
        write(f"Time for this epoch: {epoch_time}")