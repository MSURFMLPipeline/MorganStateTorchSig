""" Functions practice
- def is how you declare a function
- blah is the function name
- (string="lucas") is the parameter of the function
- you can put conditionals, print statements, and other functions in the body of the function 
- return ends the function but isnt a requirement, used for being able to call back variables, values etc.
"""
def blah(string="Lucas"): 
    print(f"My name is " + string)
    return 
blah()

def name(string="Zim"):
    print(f"Hello "+string)
    return
name()

def frequency(frequency=int):
    print(f"The input frequency is {frequency}")
    return
frequency(90)

""" Having the user input frequency""" 
def frequency2():
    f=int(input("Your user input frequency is: "))
frequency2()

import numpy as np
import math
import matplotlib.pyplot as plt


def sine_wave(phase, frequency):
    t = np.arange(0, 12, 0.1)
    Y = np.sin(frequency*t + np.radians(phase))
    print(f"The value of y is {Y}")
    print(f"The times for the sine wave are {t}")
    return Y,t


def new_sine_wave(phase, frequency, amplitude):
    #t2 = np.arange(0, 10, 0.1)
    #new_sin_wave_Y = amplitude * np.sin((frequency * t2) + np.radians(phase))
    sine_wave_Y,t= sine_wave(phase,frequency)
    sine_wave_Y=sine_wave_Y*amplitude
    return sine_wave_Y,t

sine,t= new_sine_wave(phase=180, frequency=4, amplitude=2)


plt.plot(t, sine)
plt.title("Sine Wave w/o added amplitude")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude (Y)")
plt.grid(True)
plt.show()

sine,t = new_sine_wave(30, 20, 4)
#t,sine_wave_Y=new_sine_wave(30,20,4)

plt.plot(t, sine,color='red')
plt.title("Sine Wave s2 with added amplitude")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude (Y)")
plt.grid(True)
plt.show()
