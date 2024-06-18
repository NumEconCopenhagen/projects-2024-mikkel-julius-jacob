import numpy as np

class Questien2:
    # Parameters
    alpha = 1/3
    beta = 2/3

    # Utility functions
    @staticmethod
    def utility_A(x1, x2):
        return x1**Questien2.alpha * x2**(1 - Questien2.alpha)

    @staticmethod
    def utility_B(x1, x2):
        return x1**Questien2.beta * x2**(1 - Questien2.beta)

    # Demand functions
    @staticmethod
    def demand_A(p, omega):
        p1, p2 = p
        x1_star = Questien2.alpha * (p1 * omega[0] + p2 * omega[1]) / p1
        x2_star = (1 - Questien2.alpha) * (p1 * omega[0] + p2 * omega[1]) / p2
        return np.array([x1_star, x2_star])

    @staticmethod
    def demand_B(p, omega):
        p1, p2 = p
        x1_star = Questien2.beta * (p1 * omega[0] + p2 * omega[1]) / p1
        x2_star = (1 - Questien2.beta) * (p1 * omega[0] + p2 * omega[1]) / p2
        return np.array([x1_star, x2_star]) 

    # Market clearing errors
    @staticmethod
    def market_clearing_error(p, omega_A, omega_B):
        demand_A1 = Questien2.demand_A(p, omega_A)
        demand_B1 = Questien2.demand_B(p, omega_B)
        error1 = demand_A1[0] + demand_B1[0] - (omega_A[0] + omega_B[0])
        error2 = demand_A1[1] + demand_B1[1] - (omega_A[1] + omega_B[1])
        return np.array([error1, error2])

class Question3:
    # Constants
    alpha = 1/3
    beta = 2/3
    omega_A1 = 0.8
    omega_A2 = 0.3
    omega_B1 = 1 - omega_A1
    omega_B2 = 1 - omega_A2
    p2 = 1  # Numeraire

    @staticmethod
    def excess_demand_x1(p1):
        alpha = Question3.alpha
        beta = Question3.beta
        omega_A1 = Question3.omega_A1
        omega_A2 = Question3.omega_A2
        omega_B1 = Question3.omega_B1
        omega_B2 = Question3.omega_B2
        p2 = Question3.p2
        
        xA_star_1 = alpha * (p1 * omega_A1 + p2 * omega_A2) / p1
        xB_star_1 = beta * (p1 * omega_B1 + p2 * omega_B2) / p1
        return xA_star_1 + xB_star_1 - (omega_A1 + omega_B1)

    @staticmethod
    def excess_demand_x2(p1):
        alpha = Question3.alpha
        beta = Question3.beta
        omega_A1 = Question3.omega_A1
        omega_A2 = Question3.omega_A2
        omega_B1 = Question3.omega_B1
        omega_B2 = Question3.omega_B2
        p2 = Question3.p2
        
        xA_star_2 = (1 - alpha) * (p1 * omega_A1 + p2 * omega_A2) / p2
        xB_star_2 = (1 - beta) * (p1 * omega_B1 + p2 * omega_B2) / p2
        return xA_star_2 + xB_star_2 - (omega_A2 + omega_B2)

    @staticmethod
    def total_excess_demand(p1):
        ed1 = Question3.excess_demand_x1(p1)
        ed2 = Question3.excess_demand_x2(p1)
        return abs(ed1) + abs(ed2)

class ExchangeEconomyClass:
    def __init__(self, alpha, beta, w1A, w2A):
        self.alpha = alpha
        self.beta = beta
        self.w1A = w1A
        self.w2A = w2A
        self.w1B = 1 - w1A
        self.w2B = 1 - w2A

    def utility_A(self, x1A, x2A):
        return x1A**self.alpha * x2A**(1 - self.alpha)

    def demand_A(self, p1):
        income_A = p1 * self.w1A + self.w2A  # Assuming p2 (price of good 2) is normalized to 1
        return self.alpha * income_A / p1, (1 - self.alpha) * income_A

    def demand_B(self, p1):
        income_B = p1 * self.w1B + self.w2B  # Assuming p2 (price of good 2) is normalized to 1
        return self.beta * income_B / p1, (1 - self.beta) * income_B

    def find_optimal_allocation(economy, P1):
        max_utility = -np.inf
        optimal_price = None
        optimal_allocation = None

        for p1 in P1:
            demandB = economy.demand_B(p1)
            x1A = 1 - demandB[0]
            x2A = 1 - demandB[1]
            utility = economy.utility_A(x1A, x2A)

            if utility > max_utility:
                max_utility = utility
                optimal_price = p1
                optimal_allocation = (x1A, x2A)

        return optimal_price, optimal_allocation