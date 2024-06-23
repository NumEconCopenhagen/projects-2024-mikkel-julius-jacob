# Define parameters
import numpy as np
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve

# Define parameters
alpha = 0.3
nu = 1.0
epsilon = 2.0
kappa = 0.1
gamma = 0.5
A = 1.0
p1 = 1.0
p2 = 1.0
w = 1.0
# We ignored the .par term in this question


tau_initial_guess = 0.0
# Function to calculate optimal labor
def ell_star(p1, p2, tau):
    ell1 = (p1 * A * gamma / w) ** (1 / (1 - gamma))
    ell2 = (p2 * A * gamma / w) ** (1 / (1 - gamma))
    return ell1 + ell2

# Function to calculate T given optimal tau
def calculate_T(optimal_tau):
    ell_star_val = ell_star(p1, p2, optimal_tau)
    pi1_star = (1 - gamma) / gamma * w * (p1 * A * gamma / w) ** (1 / (1 - gamma))
    pi2_star = (1 - gamma) / gamma * w * (p2 * A * gamma / w) ** (1 / (1 - gamma))

    T = optimal_tau * (1 - alpha) * (w * ell_star_val + pi1_star + pi2_star) / (p2 + optimal_tau)
    return T

# Define the objective function (negative SWF to maximize)
def objective(tau):
    ell_star_val = ell_star(p1, p2, tau)
    pi1_star = (1 - gamma) / gamma * w * (p1 * A * gamma / w) ** (1 / (1 - gamma))
    pi2_star = (1 - gamma) / gamma * w * (p2 * A * gamma / w) ** (1 / (1 - gamma))

    T_val = tau * (1 - alpha) * (w * ell_star_val + pi1_star + pi2_star) / (p2 + tau)
    c1_star = alpha * (w * ell_star_val + T_val + pi1_star + pi2_star) / p1
    c2_star = (1 - alpha) * (w * ell_star_val + T_val + pi1_star + pi2_star) / (p2 + tau)

    SWF = np.log(c1_star ** alpha * c2_star ** (1 - alpha)) - nu * (ell_star_val ** (1 + epsilon)) / (1 + epsilon) - kappa * c2_star
    return -SWF  # negative because we are maximizing

# Optimize tau
result = minimize(objective, tau_initial_guess, method='Nelder-Mead')
optimal_tau = result.x[0]

# Calculate T using the optimal tau
optimal_T = calculate_T(optimal_tau)

# Verify the results make sense
ell_star_val = ell_star(p1, p2, optimal_tau)
pi1_star = (1 - gamma) / gamma * w * (p1 * A * gamma / w) ** (1 / (1 - gamma))
pi2_star = (1 - gamma) / gamma * w * (p2 * A * gamma / w) ** (1 / (1 - gamma))

# Compute optimal consumption
T_val = optimal_tau * (1 - alpha) * (w * ell_star_val + pi1_star + pi2_star) / (p2 + optimal_tau)
c1_star = alpha * (w * ell_star_val + T_val + pi1_star + pi2_star) / p1
c2_star = (1 - alpha) * (w * ell_star_val + T_val + pi1_star + pi2_star) / (p2 + optimal_tau)

# Compute SWF
SWF = np.log(c1_star ** alpha * c2_star ** (1 - alpha)) - nu * (ell_star_val ** (1 + epsilon)) / (1 + epsilon) - kappa * c2_star

