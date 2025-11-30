import numpy as np
import matplotlib.pyplot as plt
import math
import random
"""Creating classes for simple sine wave"""
class one:
    waveform="sine" # attribute

    def __init__(self,phase,data_points,frequency):
        self.phase=phase 
        self.data_points=data_points
        self.frequency=frequency
first_wave=one(0,100,200)
second_wave=one(0,100,200)

print(one.waveform) 
print(first_wave.phase)
print(second_wave.phase)
print(second_wave.data_points)

class two(one):
    def amplitude(self):
        return f"Amplitude of the wave with frequency {self.frequency}Hz is 1.0"
amp_wave=amplitude(0,100,440)
print(amp_wave.amplitude())