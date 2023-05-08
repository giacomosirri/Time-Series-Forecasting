# This is a Python script that first reads two files from the directory given as an argument.
# One file contains the predicted values of the labels and the other the expected values.
# Then, it plots the first 500 values of the two time series over the same graph, 
# in order to show how well the model predicts future label variables values. 
# The figure is ultimately saved on a file called 'predicted_vs_expected_graph.png'.

import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    exit(1)

with open(f'{directory}predictions.txt', 'r') as f:
    next(f)
    next(f)
    values = []
    for line in f:
        line = line.strip()
        if line:
            values.append(float(line))
predictions = np.array(values)

with open(f'{directory}expected.txt', 'r') as f:
    next(f)
    next(f)
    values = []
    for line in f:
        line = line.strip()
        if line:
            values.append(float(line))
expected = np.array(values)

plt.plot(predictions[:500], "b-")
plt.plot(expected[:500], "r-")
plt.savefig(f'{directory}predicted_vs_expected_graph.png')