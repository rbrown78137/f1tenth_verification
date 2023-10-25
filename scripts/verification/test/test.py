import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # Create a grid of X and Y values
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    # Create the 2D Gaussian distribution
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    Z1 = np.exp(-0.5 * (np.square((X - mean1[0]) / cov1[0][0]) + np.square((Y - mean1[1]) / cov1[1][1])))

    # Set a fixed z-coordinate value (z=1) for the entire plot
    Z_fixed = np.ones_like(X)
    Z_fixed_minus_1 = np.ones_like(X) - 1

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # color = plt.cm.viridis(Z1)
    cmap = plt.get_cmap("viridis")
    colors = cmap(Z1)
    colors[...,3] = Z1>0.2
    # Plot the surface with variable color based on Z1
    surface = ax.plot_surface(X, Z_fixed, Y, facecolors=colors)

    surface = ax.plot_surface(X, Z_fixed_minus_1, Y, facecolors=colors)
    # Customize labels and titles as needed

    # Create a colorbar to show the correspondence between color and Z1 values
    cbar = fig.colorbar(surface, shrink=0.5, aspect=10)
    cbar.set_label('Z1 Values')

    # Show the plot
    plt.show()
    debug_var = 0