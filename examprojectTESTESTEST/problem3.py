# Define parameters
import numpy as np
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve

#q1
# Set a random seed for reproducibility
np.random.seed(42)

# Generate random points in the unit square
X = np.random.uniform(0, 1, (50, 2))

# Define the point y
y = np.random.uniform(0, 1, (2,))

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

#q2
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

# Compute barycentric coordinates for y with respect to ABC
# Define the variables A, B, C, and D
A, B, C, D = find_closest_points(X, y)

# Compute barycentric coordinates for y with respect to ABC
lambda1_ABC, lambda2_ABC, lambda3_ABC = barycentric_coordinates(y, A, B, C)

# Compute barycentric coordinates for y with respect to CDA
lambda1_CDA, lambda2_CDA, lambda3_CDA = barycentric_coordinates(y, C, D, A)

# Check if y is inside triangle ABC
is_inside_ABC = (0 <= lambda1_ABC <= 1) and (0 <= lambda2_ABC <= 1) and (0 <= lambda3_ABC <= 1)

# Check if y is inside triangle CDA
is_inside_CDA = (0 <= lambda1_CDA <= 1) and (0 <= lambda2_CDA <= 1) and (0 <= lambda3_CDA <= 1)

#q3
# Define the function f
def f(x):
    return x[0] * x[1]

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

