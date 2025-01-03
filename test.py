import numpy as np
from mytrading import SMABackTester
from ga.evolution import Evolution

import warnings
warnings.filterwarnings("ignore")

def fitness(x, y):
    sma = SMABackTester("EURUSD=X", "1990-01-01", "2025-01-01", x, y)
    return sma.get_performance()

def main():
    ind_parameters = {'lower_bound': 10, 'upper_bound': 250, 'number_of_genes': 2}
    pop_parameters = {
        'n_parents': 6,
        'offspring_size':(2, ind_parameters['number_of_genes']),
        'mutation_mean': 1,
        'mutation_sd': 1,
        'size': 10
    }

    evo = Evolution(pop_parameters, ind_parameters, fitness)
    epochs = 10000
    history = []
    x_history = []
    y_history = []

    for _ in range(epochs):
        print('Epoch {}/{}, Progress: {}%\r'.format(_+1, epochs, np.round(((_+1)/epochs)*100, 2)), end="")
        evo.step()
        history.append(evo._best_score)
        x_history.append(evo._best_individual[0][0])
        y_history.append(evo._best_individual[0][1])

    print('\nResults:')
    print('Best individual:', evo.solution.best_individual)
    print('Fitness value of best individual:', evo.solution.best_score)

if __name__ == "__main__":
    main()
