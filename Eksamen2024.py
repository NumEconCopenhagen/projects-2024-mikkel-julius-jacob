import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
import numpy as np
from types import SimpleNamespace

def question_2_1():
    # Set the seed for reproducibility
    np.random.seed(42)

    par = SimpleNamespace()
    par.J = 3
    par.N = 10
    par.K = 10000

    par.F = np.arange(1, par.N + 1)
    par.sigma = 2

    par.v = np.array([1, 2, 3])
    par.c = 1

    # Initialize arrays to store utilities
    expected_utilities = np.zeros(par.J)
    realized_utilities = np.zeros((par.N, par.J))

    # Simulate utilities
    for j in range(par.J):
        # Simulate K draws for each career track j
        epsilons = np.random.normal(0, par.sigma, par.K)
        utilities = par.v[j] + epsilons
        
        # Calculate expected utility for career track j
        expected_utilities[j] = np.mean(utilities)
        
        # Calculate realized utility for each graduate i for career track j
        for i in range(par.N):
            realized_utilities[i, j] = par.v[j] + np.random.normal(0, par.sigma)

    # Calculate average realized utility for each career track
    average_realized_utilities = np.mean(realized_utilities, axis=0)

    return expected_utilities, average_realized_utilities

def simulate_graduate_choices(par):
    choices = np.zeros(par.N, dtype=int)
    prior_expected_utilities = np.zeros(par.N)
    realized_utilities = np.zeros(par.N)
    
    for i in range(par.N):
        Fi = par.F[i]  # Number of friends for graduate i
        friends_utilities = np.zeros((Fi, par.J))
        
        for j in range(par.J):
            for f in range(Fi):
                friends_utilities[f, j] = par.v[j] + np.random.normal(0, par.sigma)
        
        prior_expected_utility = np.mean(friends_utilities, axis=0)
        noise = np.random.normal(0, par.sigma, par.J)
        expected_utility = prior_expected_utility + noise
        
        choices[i] = np.argmax(expected_utility)
        prior_expected_utilities[i] = prior_expected_utility[choices[i]]
        realized_utilities[i] = par.v[choices[i]] + np.random.normal(0, par.sigma)
    
    return choices, prior_expected_utilities, realized_utilities

def question_2_2():
    # Set the seed for reproducibility
    np.random.seed(42)

    # Define parameters
    par = SimpleNamespace()
    par.J = 3  # number of career tracks
    par.N = 10  # number of graduates
    par.sigma = 2  # standard deviation of the noise term
    par.v = np.array([1, 2, 3])  # known values of each career track
    par.F = np.arange(1, par.N + 1)  # number of friends for each graduate

    # Simulate the graduates' choices
    choices, prior_expected_utilities, realized_utilities = simulate_graduate_choices(par)

    return choices, prior_expected_utilities, realized_utilities

def simulate_initial_choices(par):
    choices = np.zeros(par.N, dtype=int)
    prior_expected_utilities = np.zeros(par.N)
    realized_utilities = np.zeros(par.N)
    
    for i in range(par.N):
        Fi = par.F[i]  # Number of friends for graduate i
        friends_utilities = np.zeros((Fi, par.J))
        
        for j in range(par.J):
            for f in range(Fi):
                friends_utilities[f, j] = par.v[j] + np.random.normal(0, par.sigma)
        
        prior_expected_utility = np.mean(friends_utilities, axis=0)
        noise = np.random.normal(0, par.sigma, par.J)
        expected_utility = prior_expected_utility + noise
        
        choices[i] = np.argmax(expected_utility)
        prior_expected_utilities[i] = prior_expected_utility[choices[i]]
        realized_utilities[i] = par.v[choices[i]] + np.random.normal(0, par.sigma)
    
    return choices, prior_expected_utilities, realized_utilities

def simulate_new_choices(par, initial_choices, realized_utilities):
    new_choices = np.zeros(par.N, dtype=int)
    new_expected_utilities = np.zeros(par.N)
    new_realized_utilities = np.zeros(par.N)
    switch_decisions = np.zeros(par.N, dtype=bool)
    
    for i in range(par.N):
        current_utility = realized_utilities[i]
        other_utilities = np.array([par.v[j] + np.random.normal(0, par.sigma) - par.c for j in range(par.J) if j != initial_choices[i]])
        all_utilities = np.insert(other_utilities, initial_choices[i], current_utility)
        
        new_choice = np.argmax(all_utilities)
        new_choices[i] = new_choice
        new_expected_utilities[i] = all_utilities[new_choice]
        new_realized_utilities[i] = par.v[new_choice] + np.random.normal(0, par.sigma)
        switch_decisions[i] = new_choice != initial_choices[i]
    
    return new_choices, new_expected_utilities, new_realized_utilities, switch_decisions

def question_2_3():
    # Set the seed for reproducibility
    np.random.seed(42)

    # Define parameters
    par = SimpleNamespace()
    par.J = 3  # number of career tracks
    par.N = 10  # number of graduates
    par.sigma = 2  # standard deviation of the noise term
    par.v = np.array([1, 2, 3])  # known values of each career track
    par.c = 1  # switching cost
    par.F = np.arange(1, par.N + 1)  # number of friends for each graduate

    # Initial choices
    initial_choices, _, initial_realized_utilities = simulate_initial_choices(par)

    # New choices with switching cost
    new_choices, new_expected_utilities, new_realized_utilities, switch_decisions = simulate_new_choices(par, initial_choices, initial_realized_utilities)

    return new_choices, new_expected_utilities, new_realized_utilities, switch_decisions



# Function for Question 3.1
def question_3_1():
    # Set the seed for reproducibility
    np.random.seed(42)

    # Generate random points in the unit square
    X = np.random.uniform(size=(50, 2))
    y = np.random.uniform(size=(2,))

    # Function to find the closest point satisfying the condition
    def find_point(X, y, condition):
        distances = np.sqrt(np.sum((X - y) ** 2, axis=1))
        valid_points = X[condition(X, y)]
        if len(valid_points) == 0:
            return None
        return valid_points[np.argmin(distances[condition(X, y)])]

    # Conditions for points A, B, C, D
    condition_A = lambda X, y: (X[:, 0] > y[0]) & (X[:, 1] > y[1])
    condition_B = lambda X, y: (X[:, 0] > y[0]) & (X[:, 1] < y[1])
    condition_C = lambda X, y: (X[:, 0] < y[0]) & (X[:, 1] < y[1])
    condition_D = lambda X, y: (X[:, 0] < y[0]) & (X[:, 1] > y[1])

    # Find points A, B, C, D
    A = find_point(X, y, condition_A)
    B = find_point(X, y, condition_B)
    C = find_point(X, y, condition_C)
    D = find_point(X, y, condition_D)

    # Return points for visualization
    return X, y, A, B, C, D

# Function for Question 3.2
def question_3_2(X, y, A, B, C, D):
    # Function to calculate barycentric coordinates
    def barycentric_coords(p, a, b, c):
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        lambda1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
        lambda2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
        lambda3 = 1 - lambda1 - lambda2
        return lambda1, lambda2, lambda3

    # Calculate barycentric coordinates for triangles ABC and CDA
    bary_coords_ABC = barycentric_coords(y, A, B, C) if A is not None and B is not None and C is not None else (None, None, None)
    bary_coords_CDA = barycentric_coords(y, C, D, A) if C is not None and D is not None and A is not None else (None, None, None)

    # Check if point y is inside triangles ABC or CDA
    inside_ABC = all(0 <= bc <= 1 for bc in bary_coords_ABC if bc is not None)
    inside_CDA = all(0 <= bc <= 1 for bc in bary_coords_CDA if bc is not None)

    return bary_coords_ABC, bary_coords_CDA, inside_ABC, inside_CDA

# Function for Question 3.3
def question_3_3(X, y, A, B, C, D):
    # Function to calculate barycentric coordinates
    def barycentric_coords(p, a, b, c):
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        lambda1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
        lambda2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
        lambda3 = 1 - lambda1 - lambda2
        return lambda1, lambda2, lambda3

    # Function to compute f(x, y)
    def f(x, y):
        return x**2 + y**2

    # Compute true value of f(y)
    true_value = f(y[0], y[1])

    # Compute barycentric coordinates for triangles ABC and CDA
    bary_coords_ABC = barycentric_coords(y, A, B, C) if A is not None and B is not None and C is not None else (None, None, None)
    bary_coords_CDA = barycentric_coords(y, C, D, A) if C is not None and D is not None and A is not None else (None, None, None)

    # Function to approximate f(y) using barycentric coordinates
    def approximate_f(bary_coords, vertices):
        return sum(lambda_i * f(v[0], v[1]) for lambda_i, v in zip(bary_coords, vertices))

    # Approximate f(y) using both triangles ABC and CDA
    approx_ABC = approximate_f(bary_coords_ABC, [A, B, C]) if None not in bary_coords_ABC else None
    approx_CDA = approximate_f(bary_coords_CDA, [C, D, A]) if None not in bary_coords_CDA else None

    # Determine which triangle contains y and use that approximation
    approx_value = None
    if all(0 <= bc <= 1 for bc in bary_coords_ABC if bc is not None):
        approx_value = approx_ABC
    elif all(0 <= bc <= 1 for bc in bary_coords_CDA if bc is not None):
        approx_value = approx_CDA

    return true_value, approx_value

# Function for Question 3.4
def question_3_4(X, points_Y):
    def find_point(X, y, condition):
        distances = np.sqrt(np.sum((X - y) ** 2, axis=1))
        valid_points = X[condition(X, y)]
        if len(valid_points) == 0:
            return None
        return valid_points[np.argmin(distances[condition(X, y)])]
    
    def barycentric_coords(p, a, b, c):
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        lambda1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
        lambda2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
        lambda3 = 1 - lambda1 - lambda2
        return lambda1, lambda2, lambda3
    
    def f(x, y):
        return x**2 + y**2
    
    true_values = []
    approximated_values = []
    
    for y in points_Y:
        condition_A = lambda X, y: (X[:, 0] > y[0]) & (X[:, 1] > y[1])
        condition_B = lambda X, y: (X[:, 0] > y[0]) & (X[:, 1] < y[1])
        condition_C = lambda X, y: (X[:, 0] < y[0]) & (X[:, 1] < y[1])
        condition_D = lambda X, y: (X[:, 0] < y[0]) & (X[:, 1] > y[1])
        
        A = find_point(X, y, condition_A)
        B = find_point(X, y, condition_B)
        C = find_point(X, y, condition_C)
        D = find_point(X, y, condition_D)
        
        true_value = f(y[0], y[1])
        true_values.append(true_value)
        
        bary_coords_ABC = barycentric_coords(y, A, B, C) if A is not None and B is not None and C is not None else (None, None, None)
        bary_coords_CDA = barycentric_coords(y, C, D, A) if C is not None and D is not None and A is not None else (None, None, None)
        
        def approximate_f(bary_coords, vertices):
            return sum(lambda_i * f(v[0], v[1]) for lambda_i, v in zip(bary_coords, vertices))
        
        approx_ABC = approximate_f(bary_coords_ABC, [A, B, C]) if None not in bary_coords_ABC else None
        approx_CDA = approximate_f(bary_coords_CDA, [C, D, A]) if None not in bary_coords_CDA else None
        
        approx_value = None
        if all(0 <= bc <= 1 for bc in bary_coords_ABC if bc is not None):
            approx_value = approx_ABC
        elif all(0 <= bc <= 1 for bc in bary_coords_CDA if bc is not None):
            approx_value = approx_CDA
        
        approximated_values.append(approx_value)
    
    # Filter out None values
    valid_indices = [i for i, v in enumerate(approximated_values) if v is not None]
    true_values = [true_values[i] for i in valid_indices]
    approximated_values = [approximated_values[i] for i in valid_indices]
    
    return true_values, approximated_values