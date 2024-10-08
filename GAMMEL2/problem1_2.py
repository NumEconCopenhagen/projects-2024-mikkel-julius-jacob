# Define parameters
import numpy as np
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve

	
# Define parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0

# Define functions
def optimal_labor(w, p, A, gamma):
    return (p * A**gamma / w)**(1 / (1 - gamma))

def optimal_output(A, l, gamma):
    return A * l**gamma

def firm_profit(w, p, A, gamma):
    return w * (p * A**gamma / w)**(1 - gamma) * (1 - gamma)

def consumer_utility(p1, p2, w, T, pi1, pi2, alpha, nu, epsilon):
    c1 = alpha * (w + T + pi1 + pi2) / p1
    c2 = (1 - alpha) * (w + T + pi1 + pi2) / (p2 + par.tau)
    l = ((c1**alpha * c2**(1 - alpha))**(1 / nu))**(1 / (1 + epsilon))
    return c1, c2, l

# Market clearing conditions
def market_clearing_conditions(prices, w, par):
    p1, p2 = prices
    
    # Firm 1
    l1 = optimal_labor(w, p1, par.A, par.gamma)
    y1 = optimal_output(par.A, l1, par.gamma)
    pi1 = firm_profit(w, p1, par.A, par.gamma)
    
    # Firm 2
    l2 = optimal_labor(w, p2, par.A, par.gamma)
    y2 = optimal_output(par.A, l2, par.gamma)
    pi2 = firm_profit(w, p2, par.A, par.gamma)
    
    # Consumer
    c1, c2, l = consumer_utility(p1, p2, w, par.T, pi1, pi2, par.alpha, par.nu, par.epsilon)
    
    # Market clearing
    labor_clearing = l1 + l2 - l
    good1_clearing = y1 - c1
    
    return [labor_clearing, good1_clearing]  # Check only two conditions


# Given equilibrium prices
p1 = 0.7475486222511643
p2 = 0.7411786704525158

# Parameters
alpha = 0.5
gamma = 0.5
A = 1
w = 1  # numeraire

# Function to calculate optimal labor allocation
def calculate_l_star(p1, p2, A, gamma, w):
    ell1_star = (p1 * A * gamma / w) ** (1 / (1 - gamma))
    ell2_star = (p2 * A * gamma / w) ** (1 / (1 - gamma))
    return ell1_star, ell2_star

# Function to calculate optimal outputs
def calculate_y_star(ell1_star, ell2_star, A, gamma):
    y1_star = A * (ell1_star ** gamma)
    y2_star = A * (ell2_star ** gamma)
    return y1_star, y2_star

# Function to calculate profits
def calculate_profits(p1, p2, A, gamma, w):
    pi1_star = (1 - gamma) / gamma * w * (p1 * A * gamma / w) ** (1 / (1 - gamma))
    pi2_star = (1 - gamma) / gamma * w * (p2 * A * gamma / w) ** (1 / (1 - gamma))
    return pi1_star, pi2_star
