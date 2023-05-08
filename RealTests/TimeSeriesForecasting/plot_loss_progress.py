import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    exit(1)

with open(f'{directory}loss-progress.txt', 'r') as f:
    next(f)
    values = []
    for line in f:
        line = line.split(":")[1].strip()
        if line:
            values.append(float(line))
loss = np.array(values)

plt.plot(loss, "b-")
plt.savefig(f'{directory}loss-progress.png')