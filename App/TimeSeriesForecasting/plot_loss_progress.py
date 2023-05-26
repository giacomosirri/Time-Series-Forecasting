# This is a Python script that plots the progress of the loss during the training of the model.
# The figure is saved on a file called 'loss_progress_graph.png'.

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    directory = sys.argv[1]
    file_name = "loss_progress_graph.png"
else:
    exit(1)

with open(os.path.join(directory, "loss_progress.txt"), 'r') as f:
    next(f)
    values = []
    for line in f:
        line = line.split(":")[1].strip()
        if line:
            values.append(float(line))
loss = np.array(values)

plt.plot(loss, "b-")
plt.savefig(os.path.join(directory, file_name))