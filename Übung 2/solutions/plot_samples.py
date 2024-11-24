#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Create 100 equidistant points between 0 and 2*pi
xs = np.linspace(0, 2. * np.pi, 100)

# Create a title for the plot
plt.figure()
plt.title("How to make plots")
# Create labels for the axes
# Note that in the label, you can use TeX-strings!
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
# Now create two plots of different functions in a single figure
plt.plot(xs, np.sin(xs), 'o-', label="$f(x)=\sin(x)$")
plt.plot(xs, np.sin(xs)**2, '*:', label="$f(x)=\sin^2(x)$")
# Now create the legend box
plt.legend()

# Create a second plot in loglog scale
plt.figure()
plt.title("A second plot, this time in loglog scale")
plt.loglog(xs, xs**2, 'o-', label="$f(x)=x^2$")
plt.loglog(xs, xs**0.5, '*:', label="$f(x)=\sqrt{x}$")
plt.legend()

# Create a third plot with four subplots
plt.figure("A title for the window")
# Select a subplot
# '221' describes the number of subplots in the figure and which subplot to select
# '22' tells to make a grid of 2 by 2 subplots
# '1' tells to select the first plot
plt.subplot(221, title="$\sin(x)$")
plt.plot(xs, np.sin(xs))
plt.subplot(222, title="$\cos(x)$")
plt.plot(xs, np.cos(xs))
plt.subplot(223, title="$\sin^2(x)$")
plt.plot(xs, np.sin(xs)**2)
plt.subplot(224, title="$\cos^2(x)$")
plt.plot(xs, np.cos(xs)**2)
plt.show()
