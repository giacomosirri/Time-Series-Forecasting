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

plt.plot(predictions[:24], "bo-", label="predictions")
plt.plot(expected[:24], "rx", label="known values")
plt.title("Predicted vs expected values for the first day of the test set.")
plt.legend(loc='upper right', fontsize='medium', frameon=True)
plt.savefig(f'{directory}predicted_vs_expected_1day.png')

days = 20
plt.plot(predictions[-24*days:], "b-", label="predictions")
plt.plot(expected[-24*days:], "r-", label="known values")
plt.title(f"Predicted vs expected trend for the last {days} days of the test set.")
plt.legend(loc='upper right', fontsize='medium', frameon=True)
plt.savefig(f'{directory}predicted_vs_expected_long_run.png')