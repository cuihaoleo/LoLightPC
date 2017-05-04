#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
import itertools
import math

readPattern = open(sys.argv[1]).read().strip()
standardPattern = sys.argv[2].strip()

chained = itertools.cycle(standardPattern)
filled = [next(chained) for i in readPattern]

minError = len(readPattern)

for i in range(len(standardPattern)):

    count = 0

    for j, bit in enumerate(readPattern):
        if bit != filled[j]:
            count += 1

    if count < minError:
        minError = count

    filled.pop(0)
    filled.append(next(chained))

print("Error: %d" % minError)
