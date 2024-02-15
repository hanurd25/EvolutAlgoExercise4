# This is a sample Python script.
import numpy as np
import random

randVer1 = random.uniform(-50, 50)
randVer2 = random.uniform(-50, 50)
expo1 = random.uniform(-5, 5)
expo2 = random.uniform(-5, 5)
expo3 = random.uniform(-5, 5)

def myFitnessFunction(a, b, c):
    return (a**2)**expo1 * ((b - randVer1)**expo2) * ((c + randVer2)**(expo3))

def initializePopulation(populationSize):
    population = []
    for _ in range(populationSize):
        a = random.uniform(-50, 50)
        b = random.uniform(-50, 50)
        c = random.uniform(-50, 50)
        population.append((a, b, c))
    return population

def calculateFitness(population):
    return [myFitnessFunction(*individual) for individual in population]

def selection(population, fitnessValues):
    # Simple tournament selection
    selectedParents = []
    for _ in range(len(population) // 2 * 2):  # Ensure an even number of parents
        tournamentSize = 3
        tournamentIndices = random.sample(range(len(population)), tournamentSize)
        tournamentFitness = [fitnessValues[i] for i in tournamentIndices]
        selectedParents.append(population[tournamentIndices[np.argmax(tournamentFitness)]])
    return selectedParents

def crossover(parent1, parent2):
    # Simple one-point crossover
    crossoverPoint = random.randint(1, len(parent1) - 1)
    child1 = tuple(np.clip(parent1[:crossoverPoint] + parent2[crossoverPoint:], -50, 50))
    child2 = tuple(np.clip(parent2[:crossoverPoint] + parent1[crossoverPoint:], -50, 50))
    return child1, child2

def mutation(individual):
    # Simple mutation by randomly perturbing one gene
    mutationPoint = random.randint(0, len(individual) - 1)
    perturbation = random.uniform(-0.5, 0.5)
    mutatedValue = np.clip(individual[mutationPoint] + perturbation, -50, 50)
    individual = list(individual)
    individual[mutationPoint] = mutatedValue
    return tuple(individual)

if __name__ == '__main__':
    population_size = 50
    generations = 100

    population = initializePopulation(population_size)

    for generation in range(generations):
        fitnessValues = calculateFitness(population)

        # Select parents
        parents = selection(population, fitnessValues)

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
        fitnessValues = calculateFitness(population)

        # Optionally, print or store the best solution and its fitness value
        bestSolution = population[np.argmax(fitnessValues)]
        bestFitness = max(fitnessValues)

        print(f"Generation {generation + 1}: Best Solution {bestSolution}, Best Fitness {bestFitness}")
    print(f"In this case, the function was f(a,b,c) = (a**2)^{expo1} * ((b - {randVer1})^{expo2} * (c + {randVer2})^{expo3} .")