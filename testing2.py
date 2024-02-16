# This is a sample Python script.
import numpy as np

import random
#changes
#randVer1 = random.uniform(0, 50)
#randVer2 = random.uniform(0, 50)
#expo1 = random.randint(0, 2)
#expo2 = random.randint(0, 2)
#expo3 = random.randint(0, 2)

populationSize = 5
generations = 500000

randVer1 = 29.6878887
randVer2 = 2.18913
expo1 = 2
expo2 = 2
expo3 = 2
probabilityList = [0.5, 0.5, 0.5]

bestFitness = 0.0

#I dont want to potentially do the root of a negative number
def myFitnessFunction(paramsABC):
    return ((paramsABC[0]**2)**expo1) + ((paramsABC[1] + randVer1)**expo2) + ((paramsABC[2] + randVer2)**(expo3))

def biasedRandomOutput(): #beta distrobution is the goated one
    #Shape parameters for the beta distribution
    alpha = 0.7
    beta = 0.7
    return np.random.beta(alpha, beta) * 50  # Scaling to the range to be from 0 to  50


def initializePopulation(populationSize): #returning a list of the population
    population = []
    for _ in range(populationSize):
        population.append([biasedRandomOutput(), biasedRandomOutput(), biasedRandomOutput()])
    return population

def calculateFitness(population):
    return [myFitnessFunction(individual) for individual in population]

def selection(population, fitnessValues):
    return random.choices(population, weights=fitnessValues, k=2)


def crossover(parent1, parent2, probabilityList):
    child1 = []
    for gene1, gene2, prob in zip(parent1, parent2, probabilityList):
        if random.uniform(0, 1) < prob:
            child1.append(gene2)
        else:
            child1.append(gene1)
    return child1


def mutation(individual):
    # Simple mutation by randomly perturbing one gene
    i = 0
    for i in range(0, len(individual)):
        if 0.05 > random.uniform(0, 1):

            individual[i] = biasedRandomOutput()
        #if (individual[i] + randomVariable > 50):
        #    individual[i] = random.uniform(0, 50)
        #else:
        #    individual[i] = individual[i] + random.uniform(0, 1) # child1
    return individual

if __name__ == '__main__':
    population = initializePopulation(populationSize)

    for generation in range(generations):
        fitnessValues = calculateFitness(population)

        parent1, parent2 = selection(population, fitnessValues)


        offspring = [crossover(parent1, parent2, probabilityList)]


        #Might be handy if i want to create multiple children for each iteration
        offspring = [mutation(individual) for individual in offspring]

        population = population + offspring

        # Replace worst performing individuals with new children
        for _ in range(0, len(offspring)):  # the worst performer for each itiration with be replaced with a new child
            worstIndex = np.argmin(fitnessValues)  # finding the index of the worst performing induvidual
            del population[worstIndex]  # using del to performe removal of the worst permorming
            del fitnessValues[worstIndex]


        fitnessValues = calculateFitness(population)

        # Get indices of worst individuals
        #worst_indices = np.argsort(new_fitness_values)[:2]


        # Update population and fitness values

        best_index = np.argmax(fitnessValues)
        best_solution = population[best_index]
        best_fitness = fitnessValues[best_index]
        print(f" \n The length of the population is {len(population)} \n")
        print(f"this is the worst performer {population[np.argmin(fitnessValues)]}")
        print(f"Generation {generation + 1}: Best Solution {best_solution}")
        print(f"The best fitness is {best_fitness}")

    print(f"In this case, the function was f(a,b,c) = (a**2)^{expo1} * ((b + {randVer1})^{expo2} * (c + {randVer2})^{expo3} .")