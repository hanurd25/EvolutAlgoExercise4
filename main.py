##Declaration of sources:
#The lecture notes from part B in 'Intelligente Systemer'

#Insperation of crossover methods: https://medium.com/@samiran.bera/crossover-operator-the-heart-of-genetic-algorithm-6c0fdcb405c0

#Inspiration for the creation of the fitness function and the mutation functin: https://medium.com/@Data_Aficionado_1083/genetic-algorithms-optimizing-success-through-evolutionary-computing-f4e7d452084f

#No code is directly copied
## 
import numpy as np
import matplotlib.pyplot as plt
import random
#changes
#randVer1 = random.uniform(0, 10)
#randVer2 = random.uniform(0, 10)
#randVer3 = random.uniform(0, 10)
#expo1 = random.randint(0, 2)
#expo2 = random.randint(0, 2)
#expo3 = random.randint(0, 2)

bestPerformers = [] #List for holding the best performer for every generation
averageFitness = []
medianFitness = []

populationSize = 1000
generations = 500
#250 generations gives a very accurate output, in the end.


randVer1 = 7.6878887
randVer2 = 40.18913
randVer3 = 2.18913
expo1 = 2
expo2 = 2
expo3 = 2
probabilityList = [0.5, 0.5, 0.5]

bestFitness = 0.0

#I dont want to potentially do the root of a negative number
def myFitnessFunction(paramsABC):
    return (-(paramsABC[0]-randVer1)**expo1 - ((paramsABC[1] - randVer2)**expo2) + ((paramsABC[2] + randVer3)**(expo3))+ 5000)

def biasedRandomOutput(): #beta distrobution is the goated one
    #Shape parameters for the beta distribution
    alpha = 0.8
    beta = 0.8
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
    for i in range(0, len(individual)):                                  # There is a 5 percent chance of overwriting a value inside the list with a new randome one
        if 0.05 > random.uniform(0, 1):

            individual[i] = biasedRandomOutput()

        elif (0.12 > random.uniform(0, 1)) and (individual[i]+1 <= 50): #There is a higher chance of a small adjustment to the genes
            individual[i] = individual[i] + random.uniform(0, 1)

        elif (0.25 > random.uniform(0, 1)) and (individual[i] + 0.25 <= 50): #There is a higher chance of a small adjustment to the genes
            individual[i] = individual[i] + random.uniform(0, 0.25)

            # (individual[i] + randomVariable > 50):
        #    individual[i] = random.uniform(0, 50)
        #else:
        #    individual[i] = individual[i] + random.uniform(0, 1) # child1
    return individual

if __name__ == '__main__':
    population = initializePopulation(populationSize)

    for generation in range(generations):
        fitnessValues = calculateFitness(population)

        #Selecting parents with the fitness proportional selection method
        parent1, parent2 = selection(population, fitnessValues)

        #Creating the new offspring with the uniform crossover technique
        offspring = [crossover(parent1, parent2, probabilityList)]


        #Might be handy if i want to create multiple children for each iteration
        offspring = [mutation(individual) for individual in offspring]

        # Replace worst performing individuals with new children
        for _ in range(0, len(offspring)):  # the worst performer for each itiration with be replaced with a new child
            worstIndex = np.argmin(fitnessValues)  # finding the index of the worst performing induvidual
            del population[worstIndex]  # using del to performe removal of the worst permorming
            del fitnessValues[worstIndex]


        #adding the offspring to the population
        population = population + offspring


        fitnessValues = calculateFitness(population)


        # Update population and fitness values

        bestIndex = np.argmax(fitnessValues)
        bestSolution = population[bestIndex]
        bestFitness = fitnessValues[bestIndex]

        bestPerformers.append([generation + 1, bestFitness])
        averageFitness.append([generation + 1, sum(fitnessValues)/len(fitnessValues)])
        medianFitness.append([generation + 1, np.median(fitnessValues)])

        print(f" \n The length of the population is: {len(population)} \n")
        print(f"This is the worst performer: {population[np.argmin(fitnessValues)]}")
        print(f"Generation {generation + 1}: Best Solution {bestSolution}")
        print(f"The best fitness is {bestFitness}")


    print(f"In this case, the function was f(a,b,c) = -(a-{randVer1})^{expo1} - ((b - {randVer1})^{expo2} + (c + {randVer2})^{expo3} .")

    # Plotting
    generationsBest, fitnessBest = zip(*bestPerformers)
    generationsAvg, fitnessAvg = zip(*averageFitness)
    generationsMedian, fitnessMedian = zip(*medianFitness)

    plt.plot(generationsBest, fitnessBest, label='Best performer')
    plt.plot(generationsAvg, fitnessAvg, label='Average fitness')
    plt.plot(generationsMedian, fitnessMedian, label='Median performer')

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Comparing best performer, average fitness, and median performer')
    plt.legend()
    plt.show()
