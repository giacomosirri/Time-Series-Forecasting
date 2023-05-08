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
plt.savefig(f'{directory}predicted-vs-expected.png')