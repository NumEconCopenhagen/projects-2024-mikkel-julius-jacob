# Define parameters
from types import SimpleNamespace
import numpy as np
import pandas as pd

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