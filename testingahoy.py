#%%
import matplotlib.pyplot as plt
import numpy as np

print("Libraries imported.")

#%%
# Prepare some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

print("Data prepared.")

#%%
# Create and display the first plot
plt.figure()
plt.plot(x, y1)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

print("First plot generated.")

#%%
# Create and display the second plot
plt.figure()
plt.plot(x, y2, color='red')
plt.title("Cosine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

print("Second plot generated.")
# %%
