#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
import math


DEBUG = True
DEBUG_POS = 3

BIT_ZERO_FREQ = 6
BIT_ONE_FREQ = 4
DATARATE = math.gcd(BIT_ZERO_FREQ, BIT_ONE_FREQ)
ORIGINAL_VIDEO_FPS = BIT_ZERO_FREQ * BIT_ONE_FREQ // DATARATE
INTERMEDIATE_DATA_FPS = 4 * ORIGINAL_VIDEO_FPS


def StandardSignalSource(startFromZero = True):
    nZero = ORIGINAL_VIDEO_FPS // BIT_ZERO_FREQ
    nOne = ORIGINAL_VIDEO_FPS // BIT_ONE_FREQ
    flagBit = 0 if startFromZero else 1

    while True:
        nWhat = nZero if flagBit == 0 else nOne

        for i in range(ORIGINAL_VIDEO_FPS // DATARATE):
            highlight = i % nWhat < nWhat // 2
            for j in range(INTERMEDIATE_DATA_FPS // ORIGINAL_VIDEO_FPS):
                yield 1 if highlight else -1

        flagBit = 1 - flagBit
    
    
def findShift(signal):
    print(signal)
    source = StandardSignalSource()
    signalBuffer = []

    for i in range(signal.shape[0]):
        signalBuffer.append(next(source))

    bestCorr = -math.inf
    bestShift = 0
    bestSignal = None
    nFramesPerCycle = INTERMEDIATE_DATA_FPS // DATARATE;

    for i in range(2*nFramesPerCycle):
        result = signal.dot(signalBuffer)
        if result > bestCorr:
            bestCorr = result
            bestShift = i
            bestSignal = np.array(signalBuffer)

        signalBuffer.pop(0)
        signalBuffer.append(next(source))

    """
    plt.plot(bestSignal)
    plt.plot(signal - np.mean(signal))
    plt.show()
    """

    return nFramesPerCycle - bestShift % nFramesPerCycle


def main():
    rawData = np.loadtxt(sys.argv[1])

    timeSeries = rawData[:, 0]
    timeSeries -= np.min(timeSeries)

    brightnessMatrix = rawData[:, 1:]
    blockN = brightnessMatrix.shape[-1]
    print("blockN = %d" % blockN)

    # resample
    resampledT = np.arange(0, timeSeries.max(), 1000000.0 / INTERMEDIATE_DATA_FPS)
    resampledData = []
    for i in range(0, blockN):
        brightness = brightnessMatrix[:, i]
        brightness -= np.mean(brightness)
        func = interpolate.interp1d(timeSeries, brightnessMatrix[:, i], kind = "cubic")
        resampled = func(resampledT)
        resampledData.append(resampled)

    windowSize = INTERMEDIATE_DATA_FPS // DATARATE
    shift = findShift(resampledData[0])
    print("shift = %d" % shift)

    if DEBUG:
        debugDecoded = [0] * shift

    decodedBits = []
    freq0 = windowSize // (INTERMEDIATE_DATA_FPS // BIT_ZERO_FREQ)
    freq1 = windowSize // (INTERMEDIATE_DATA_FPS // BIT_ONE_FREQ)

    src0 = StandardSignalSource(startFromZero = True)
    src1 = StandardSignalSource(startFromZero = False)
    stdSignal0 = [next(src0) for i in range(windowSize)]
    stdSignal1 = [next(src1) for i in range(windowSize)]

    for startN in np.arange(shift, resampledT.size - windowSize + 1, windowSize):
        for pos in range(1, blockN):
            window = resampledData[pos][startN : startN + windowSize]
            window -= np.mean(window)
            conv0 = window.dot(stdSignal0)
            conv1 = window.dot(stdSignal1)

            windowFFT = np.abs(np.fft.fft(window))
            level = np.log(windowFFT[freq1] / windowFFT[freq0])

            bit = 0 if conv0 > conv1 else 1
            bitFFT = 0 if level < 0  else 1

            #bit = bitFFT
            print(bit, end="")
            decodedBits.append(bit)

            if DEBUG and pos == DEBUG_POS:
                debugDecoded.extend([bit] * windowSize)

        print()

    print("".join(str(i) for i in decodedBits))
    
    if DEBUG:
        for i in np.arange(shift, resampledT.size, windowSize):
            plt.axvline(x = i, color = "k")
     
        plt.plot(resampledData[0])
        plt.plot(resampledData[DEBUG_POS], linewidth=1.5)
        plt.plot(debugDecoded)
        plt.show()

        debugDecoded = []


if __name__ == "__main__":
    main()
