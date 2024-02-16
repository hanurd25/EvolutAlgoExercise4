# This is a sample Python script.
import numpy as np

import random
#changes
#randVer1 = random.uniform(-50, 50)
#randVer2 = random.uniform(-50, 50)
#expo1 = random.uniform(-5, 5)
#expo2 = random.uniform(-5, 5)
#expo3 = random.uniform(-5, 5)

randVer1 = -29.6878887
randVer2 = -2.18913
expo1 = 2.8887135
expo2 = -3.7088987
expo3 = 3.561440

def myFitnessFunction(paramsABC):
    return ((paramsABC[0]**2)**expo1) - ((paramsABC[1] - randVer1)**expo2) +((paramsABC[2] + randVer2)**(expo3))

def initializePopulation(populationSize):
    population = []
    for _ in range(populationSize):
        population.append([random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(-50, 50)])
    return population

def calculateFitness(population):
    return [myFitnessFunction(individual) for individual in population]

def selection(population, fitnessValues):
    # Simple tournament selection
    selectedParents = []
    for _ in range(len(population) // 2 * 2):  # Ensure an even number of parents
        tournamentSize = 3
        tournamentIndices = random.sample(range(len(population)), tournamentSize)
        tournamentFitness = [fitnessValues[i] for i in tournamentIndices]
        selectedParents.append(population[tournamentIndices[np.argmax(tournamentFitness)]])
    return selectedParents


def crossover(parent1, parent2, probabilityList):
    i= 0
    for gene1, gene2 in zip(parent1, parent2):
        if (probabilityList[i] > random.uniform(0, 1)):
            child1[i] = gene2
            child2[i] = gene1
        i= i+1
    return parent1, parent2

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
    generations = 500

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

        # Keep the best individual from the current population
        best_index = np.argmax(fitnessValues)
        offspring.append(population[best_index])

        # Update the population with the new offspring
        population = offspring

        # Recalculate fitness values for the new population
        fitnessValues = calculateFitness(population)

        # Optionally, print or store the best solution and its fitness value
        bestSolution = population[np.argmax(fitnessValues)]
        bestFitness = myFitnessFunction(bestSolution)
        print(randVer1)
        print(randVer2)
        print(expo1)
        print(expo2)
        print(expo3)
        print(f"This is just for testing {myFitnessFunction([1, 3, 4])}")
        print(f"Generation {generation + 1}: Best Solution {bestSolution}")
        print(f"The best fitness is Best Fitness {myFitnessFunction(bestSolution)}")
    print(f"In this case, the function was f(a,b,c) = (a**2)^{expo1} * ((b - {randVer1})^{expo2} * (c + {randVer2})^{expo3} .")