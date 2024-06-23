# Define parameters
import numpy as np
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve


# Set a random seed for reproducibility
np.random.seed(42)

# Generate random points in the unit square
X = np.random.uniform(0, 1, (50, 2))

# Define the point y
y = np.random.uniform(0, 1, (2,))

# Define the set Y
Y = np.array([(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.8, 0.2), (0.5, 0.5)])

# Function to find the closest point in each quadrant
def find_closest_points(X, y):
    # Initialize the points A, B, C, D with None
    A, B, C, D = None, None, None, None
    min_distances = [np.inf, np.inf, np.inf, np.inf]

    for point in X:
        x1, x2 = point
        y1, y2 = y
        distance = np.linalg.norm(point - y)

        if x1 > y1 and x2 > y2:  # Quadrant 1
            if distance < min_distances[0]:
                A = point
                min_distances[0] = distance
        elif x1 > y1 and x2 < y2:  # Quadrant 2
            if distance < min_distances[1]:
                B = point
                min_distances[1] = distance
        elif x1 < y1 and x2 < y2:  # Quadrant 3
            if distance < min_distances[2]:
                C = point
                min_distances[2] = distance
        elif x1 < y1 and x2 > y2:  # Quadrant 4
            if distance < min_distances[3]:
                D = point
                min_distances[3] = distance

    return A, B, C, D

# Find the closest points
A, B, C, D = find_closest_points(X, y)

# Function to compute barycentric coordinates
def barycentric_coordinates(P, A, B, C):
    Px, Py = P
    Ax, Ay = A
    Bx, By = B
    Cx, Cy = C
    
    denominator = (By - Cy) * (Ax - Cx) + (Cx - Bx) * (Ay - Cy)
    
    lambda1 = ((By - Cy) * (Px - Cx) + (Cx - Bx) * (Py - Cy)) / denominator
    lambda2 = ((Cy - Ay) * (Px - Cx) + (Ax - Cx) * (Py - Cy)) / denominator
    lambda3 = 1 - lambda1 - lambda2
    
    return lambda1, lambda2, lambda3

# Define the function f
def f(x):
    return x[0] * x[1]

# Compute f values at A, B, C, D
f_A = f(A)
f_B = f(B)
f_C = f(C)
f_D = f(D)

# Iterate over all points in Y
results = []

for y in Y:
    # Compute barycentric coordinates for y with respect to ABC and CDA
    lambda1_ABC, lambda2_ABC, lambda3_ABC = barycentric_coordinates(y, A, B, C)
    lambda1_CDA, lambda2_CDA, lambda3_CDA = barycentric_coordinates(y, C, D, A)
    
    # Interpolate f(y) using ABC
    f_y_ABC = lambda1_ABC * f_A + lambda2_ABC * f_B + lambda3_ABC * f_C
    
    # Interpolate f(y) using CDA
    f_y_CDA = lambda1_CDA * f_C + lambda2_CDA * f_D + lambda3_CDA * f_A
    
    # Determine which triangle y is inside based on barycentric coordinates
    if (0 <= lambda1_ABC <= 1) and (0 <= lambda2_ABC <= 1) and (0 <= lambda3_ABC <= 1):
        f_y_approx = f_y_ABC
    elif (0 <= lambda1_CDA <= 1) and (0 <= lambda2_CDA <= 1) and (0 <= lambda3_CDA <= 1):
        f_y_approx = f_y_CDA
    else:
        f_y_approx = None
    
    # Compute the true value of f(y)
    f_y_true = f(y)
    
    # Store the results
    results.append({
        'y': y,
        'f_y_approx': f_y_approx,
        'f_y_true': f_y_true,
        'error': abs(f_y_true - f_y_approx) if f_y_approx is not None else None,
        'inside_ABC': (0 <= lambda1_ABC <= 1) and (0 <= lambda2_ABC <= 1) and (0 <= lambda3_ABC <= 1),
        'inside_CDA': (0 <= lambda1_CDA <= 1) and (0 <= lambda2_CDA <= 1) and (0 <= lambda3_CDA <= 1)
    })
