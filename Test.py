# In this script I am going to write a simple code for COHERENT OPTICAL NETWORKS 





import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

# Define the sinc function
def sinc(x):
    return np.tan(x*100 )**2/x * np.log(x*np.sin(x)) * x**4  # np.sinc in NumPy is normalized as sinc(pi*x)

# Create an array of x values
x = np.linspace(-100, 100, 5000)

# Evaluate the sinc function
y = sinc(x)

# Perform Fourier Transform
y_fft = fftshift(fft(y))

# Create an array of frequency values for plotting the Fourier transform
freq = np.fft.fftfreq(x.size, x[1] - x[0])
freq = fftshift(freq)

# Plot the sinc function
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Sinc Function')
plt.xlabel('x')
plt.ylabel('sinc(x)')
plt.grid(True)

# Plot the magnitude of the Fourier transform
plt.subplot(1, 2, 2)
plt.plot(freq, np.abs(y_fft))
plt.title('Fourier Transform of Sinc Function')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
