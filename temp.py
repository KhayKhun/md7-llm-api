import matplotlib.pyplot as plt
import numpy as np

# Generate x values
x = np.linspace(-10, 10, 400)  # 400 points between -10 and 10

# Calculate y values
y = x**2

# Create the plot
plt.plot(x, y, label='y = x^2')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of y = x^2')

# Add a legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()