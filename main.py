# This is a sample Python script.
import numpy as np
import random

randVer1 = random.uniform(-50, 50)
randVer2 = random.uniform(-50, 50)
expo1 = random.uniform(-5, 5)
expo2 = random.uniform(-5, 5)
expo3 = random.uniform(-5, 5)

def myFunction(a, b, c):
    return (a**2)**expo1 * ((b - randVer1)**expo2) * ((c + randVer2)**(expo3))

def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        a = random.uniform(-50, 50)
        b = random.uniform(-50, 50)
        c = random.uniform(-50, 50)
        population.append((a, b, c))
    return population

def calculate_fitness(population):
    return [myFunction(*individual) for individual in population]

def selection(population, fitness_values):
    # Simple tournament selection
    selected_parents = []
    for _ in range(len(population) // 2 * 2):  # Ensure an even number of parents
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        selected_parents.append(population[tournament_indices[np.argmax(tournament_fitness)]])
    return selected_parents

def crossover(parent1, parent2):
    # Simple one-point crossover
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = tuple(np.clip(parent1[:crossover_point] + parent2[crossover_point:], -50, 50))
    child2 = tuple(np.clip(parent2[:crossover_point] + parent1[crossover_point:], -50, 50))
    return child1, child2

def mutation(individual):
    # Simple mutation by randomly perturbing one gene
    mutation_point = random.randint(0, len(individual) - 1)
    perturbation = random.uniform(-0.5, 0.5)
    mutated_value = np.clip(individual[mutation_point] + perturbation, -50, 50)
    individual = list(individual)
    individual[mutation_point] = mutated_value
    return tuple(individual)

if __name__ == '__main__':
    population_size = 50
    generations = 100

    population = initialize_population(population_size)

    for generation in range(generations):
        fitness_values = calculate_fitness(population)

        # Select parents
        parents = selection(population, fitness_values)

        # Create offspring through crossover
        offspring = []
        for i in range(0, len(parents), 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            offspring.append(child1)
            offspring.append(child2)

        # Apply mutation to offspring
        offspring = [mutation(individual) for individual in offspring]

        population = offspring

        # Recalculate fitness values for the new population
        fitness_values = calculate_fitness(population)

        # Optionally, print or store the best solution and its fitness value
        best_solution = population[np.argmax(fitness_values)]
        best_fitness = max(fitness_values)

        print(f"Generation {generation + 1}: Best Solution {best_solution}, Best Fitness {best_fitness}")
    print(f"In this case, the function was f(a,b,c) = (a**2)^{expo1} * ((b - {randVer1})^{expo2} * (c + {randVer2})^{expo3} .")