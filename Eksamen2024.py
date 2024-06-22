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


# Function to generate random points in the unit square
def generate_random_points(seed=42, num_points=50):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, (num_points, 2))
    y = np.random.uniform(0, 1, (2,))
    return X, y

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