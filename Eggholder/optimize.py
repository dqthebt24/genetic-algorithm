from deap import base
from deap import creator
from deap import tools

import random
import copy
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import elitism
from datetime import datetime

POPULATION = []

# problem constants:
DIMENSIONS = 2  # number of dimensions
BOUND_LOW, BOUND_UP = -512.0, 512.0  # boundaries for all dimensions

# Genetic Algorithm constants:
POPULATION_SIZE = 300
P_CROSSOVER = 0.5  # probability for crossover
P_MUTATION = 0.1   # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 500
HALL_OF_FAME_SIZE = 30
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# Define fintness trategy (min or max)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class as list, using fitness function as the definition above
creator.create("Individual", list, fitness=creator.FitnessMin)

# Function random float vector DIMENSIONS demensions with values in [low,up] 
def randomFloat(low, up):
	return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

# Eggholder function as the given individual's fitness:
def eggholder(individual):
	x = individual[0]
	y = individual[1]
	f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
	return f,  # return a tuple

# Init population
def initPopulation():
	global POPULATION

	toolbox = getToolbox()
	random.seed(datetime.now())

	# create initial population (generation 0):
	POPULATION = toolbox.populationCreator(n=POPULATION_SIZE)

def getStats():
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("min", np.min)
	stats.register("avg", np.mean)
	return stats 

def getToolbox():
	toolbox = base.Toolbox()

	# attrFloat as an operator returns random float numbers
	toolbox.register("attrFloat", randomFloat, BOUND_LOW, BOUND_UP)

	# create the individual operator to fill up an Individual instance:
	toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.attrFloat)

	# create the population operator to generate a list of individuals:
	toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

	toolbox.register("evaluate", eggholder)

	# genetic operators:
	toolbox.register("select", tools.selTournament, tournsize=2)
	toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR)
	toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=CROWDING_FACTOR,
		indpb=1.0/DIMENSIONS)
	return toolbox

def gaSolution():
	# random.seed(datetime.now())

	lPopulation = copy.deepcopy(POPULATION)

	# Get a toolbox
	toolbox = getToolbox()
	stats = getStats()
	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	# Run GA algorithm
	populationLocal, logbook = elitism.eaSimpleWithElitism(lPopulation, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, 
		ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

	minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
	best = hof.items[0]

	return best, minFitnessValues, meanFitnessValues

def agaSolution():
	# random.seed(datetime.now())

	lPopulation = copy.deepcopy(POPULATION)

	# Get a toolbox
	toolbox = getToolbox()
	stats = getStats()
	hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

	# Run AGA algorithm
	k1 = 1
	k2 = 0.5
	k3 = 1
	k4 = 0.5
	populationLocal, logbook = elitism.eaAdaptiveWithElitism(lPopulation, toolbox, 
		ngen=MAX_GENERATIONS, k1=k1, k2=k2, k3=k3, k4=k4, stats=stats, halloffame=hof, verbose=True)

	minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
	best = hof.items[0]

	return best, minFitnessValues, meanFitnessValues

def main():
	# Init population
	initPopulation()

	agaBest, agaMins, agaMeans = agaSolution()
	best, gaMins, gaMeans = gaSolution()

	print("-- Best Individual = ", best)
	print("-- Best Fitness = ", best.fitness.values[0])

	print("-- AGA Best Individual = ", agaBest)
	print("-- AGA Best Fitness = ", agaBest.fitness.values[0])

	# plot statistics
	sns.set_style("whitegrid")
	plt.plot(gaMins, color='red')
	plt.xlabel('Generation')
	plt.plot(agaMins, color='blue')
	plt.gca().legend(('GA Mins','AGA Mins'))
	plt.show()

if __name__ == "__main__":
	main()