import numpy as np
from types import SimpleNamespace

## Question 2.2
# Simulate the graduates' choices
def simulate_graduate_choices(par):
    choices = np.zeros(par.N, dtype=int)
    prior_expected_utilities = np.zeros(par.N)
    realized_utilities = np.zeros(par.N)
    
    for i in range(par.N):
        Fi = i + 1  # Number of friends
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


class Question23:
    def __init__(self, J, N, sigma, v, c):
        self.par = SimpleNamespace()
        self.par.J = J  # number of career tracks
        self.par.N = N  # number of graduates
        self.par.sigma = sigma  # standard deviation of the noise term
        self.par.v = v  # known values of each career track
        self.par.c = c  # switching cost
    
    def simulate_initial_choices(self):
        choices = np.zeros(self.par.N, dtype=int)
        prior_expected_utilities = np.zeros(self.par.N)
        realized_utilities = np.zeros(self.par.N)
        
        for i in range(self.par.N):
            Fi = i + 1  # Number of friends
            friends_utilities = np.zeros((Fi, self.par.J))
            
            for j in range(self.par.J):
                for f in range(Fi):
                    friends_utilities[f, j] = self.par.v[j] + np.random.normal(0, self.par.sigma)
            
            prior_expected_utility = np.mean(friends_utilities, axis=0)
            noise = np.random.normal(0, self.par.sigma, self.par.J)
            expected_utility = prior_expected_utility + noise
            
            choices[i] = np.argmax(expected_utility)
            prior_expected_utilities[i] = prior_expected_utility[choices[i]]
            realized_utilities[i] = self.par.v[choices[i]] + np.random.normal(0, self.par.sigma)
        
        return choices, prior_expected_utilities, realized_utilities

    def simulate_new_choices(self, initial_choices, realized_utilities):
        new_choices = np.zeros(self.par.N, dtype=int)
        new_expected_utilities = np.zeros(self.par.N)
        new_realized_utilities = np.zeros(self.par.N)
        switch_decisions = np.zeros(self.par.N, dtype=bool)
        
        for i in range(self.par.N):
            current_utility = realized_utilities[i]
            other_utilities = np.array([self.par.v[j] + np.random.normal(0, self.par.sigma) - self.par.c for j in range(self.par.J) if j != initial_choices[i]])
            all_utilities = np.insert(other_utilities, initial_choices[i], current_utility)
            
            new_choice = np.argmax(all_utilities)
            new_choices[i] = new_choice
            new_expected_utilities[i] = all_utilities[new_choice]
            new_realized_utilities[i] = self.par.v[new_choice] + np.random.normal(0, self.par.sigma)
            switch_decisions[i] = new_choice != initial_choices[i]
        
        return new_choices, new_expected_utilities, new_realized_utilities, switch_decisions
