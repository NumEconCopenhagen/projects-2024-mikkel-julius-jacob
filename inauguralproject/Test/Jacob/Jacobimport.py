from types import SimpleNamespace

import numpy as np

class ExchangeEconomyClass:
    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3

    def utility_A(self,x1A,x2A):
        pass

    def utility_B(self,x1B,x2B):
        pass

    def demand_A(self,p1):
        pass

    def demand_B(self,p1):
        pass

    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2    

    def __init__(self):
        # Initialize model parameters
        par = self.par = SimpleNamespace()
        par.alpha = 1/3  # Preference parameter for consumer A
        par.beta = 2/3   # Preference parameter for consumer B
        par.w1A = 0.8    # Initial endowment of good 1 for consumer A
        par.w2A = 0.3    # Initial endowment of good 2 for consumer A
        # Calculate initial utility levels based on the endowments
        self.uA_initial = self.utility_A(par.w1A, par.w2A)
        self.uB_initial = self.utility_B(1 - par.w1A, 1 - par.w2A)

    def utility_A(self, x1A, x2A):
        return x1A ** self.par.alpha * x2A ** (1 - self.par.alpha)

    def utility_B(self, x1B, x2B):
        return x1B ** self.par.beta * x2B ** (1 - self.par.beta)

    def find_pareto_improvements(self):
        N = 75
        x1A_grid = np.linspace(0, 1, N)
        x2A_grid = np.linspace(0, 1, N)
        pareto_set = np.zeros((N, N))
        for i, x1A in enumerate(x1A_grid):
            for j, x2A in enumerate(x2A_grid):
                x1B, x2B = 1 - x1A, 1 - x2A
                uA = self.utility_A(x1A, x2A)
                uB = self.utility_B(x1B, x2B)
                if uA >= self.uA_initial and uB >= self.uB_initial:
                    pareto_set[i, j] = 1
        return x1A_grid, x2A_grid, pareto_set

    