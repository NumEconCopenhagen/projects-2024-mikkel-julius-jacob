# Define parameters
import numpy as np
from types import SimpleNamespace
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import fsolve

par = SimpleNamespace()
par.alpha = 1.0
par.gamma = 0.5
par.nu = 0.5
par.epsilon = 1.5
par.mu = 1.0
par.tau = 0.0
par.A = 1.0
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
def market_clearing(p1, p2, w, par):
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
    good2_clearing = y2 - c2
    
    return labor_clearing, good1_clearing, good2_clearing

# Prices
p1_values = np.linspace(0.1, 2.0, 10)
p2_values = np.linspace(0.1, 2.0, 10)

# Check market clearing conditions for each combination of p1 and p2
results = []
for p1 in p1_values:
    for p2 in p2_values:
        labor_clearing, good1_clearing, good2_clearing = market_clearing(p1, p2, 1, par)
        results.append({
            'p1': p1,
            'p2': p2,
            'labor_clearing': labor_clearing,
            'good1_clearing': good1_clearing,
            'good2_clearing': good2_clearing
        })
# Create DataFrame for results
df_results = pd.DataFrame(results)

# Display DataFrame
from IPython.display import display
display(df_results)


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

# Initial guess for p1 and p2
initial_guess = [1.0, 1.0]

# Solve for market clearing prices
solution = fsolve(market_clearing_conditions, initial_guess, args=(1, par))

# Display the solution
print("Market clearing prices:")
print(f"p1 = {solution[0]}")
print(f"p2 = {solution[1]}")


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

# Calculate optimal labor allocation
ell1_star, ell2_star = calculate_l_star(p1, p2, A, gamma, w)

# Calculate optimal outputs
y1_star, y2_star = calculate_y_star(ell1_star, ell2_star, A, gamma)

# Calculate implied profits
pi1_star, pi2_star = calculate_profits(p1, p2, A, gamma, w)

# Total labor supplied
ell_star_val = ell1_star + ell2_star

# Calculate optimal consumption given T = 0
T = 0
c1_star = alpha * (w * ell_star_val + T + pi1_star + pi2_star) / p1
c2_star = (1 - alpha) * (w * ell_star_val + T + pi1_star + pi2_star) / p2


# Print results to check market clearing
print(f"Optimal labor (ell1*, ell2*): ({ell1_star}, {ell2_star})")
print(f"Optimal output (y1*, y2*): ({y1_star}, {y2_star})")
print(f"Implied profits (pi1*, pi2*): ({pi1_star}, {pi2_star})")
print(f"Total labor supplied (ell*): {ell_star_val}")
print(f"Optimal consumption (c1*, c2*): ({c1_star}, {c2_star})")

# Tolerance for approximate equality
tolerance = 1e-2

# Check market clearing conditions
labor_market_clears = np.abs(ell_star_val - (ell1_star + ell2_star)) < tolerance
goods_market_1_clears = np.abs(c1_star - y1_star) < tolerance
goods_market_2_clears = np.abs(c2_star - y2_star) < tolerance

print(f"Labor market clears: {labor_market_clears}")
print(f"Goods market 1 clears: {goods_market_1_clears}")
print(f"Goods market 2 clears: {goods_market_2_clears}")
