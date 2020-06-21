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

# Simulate stress and S, sigma_i
S = 6
L = 9.144
P = 2800
SIGMS = []
FORCES = np.array([61.540, 1.530, 5.790, 72.710, 2.660, 19.340,23.125, 83.810, 79.480, 6.503])
LENGTHS = np.array([L, L, L, L, L, L, L, np.sqrt(2)*L, np.sqrt(2)*L, np.sqrt(2)*L, np.sqrt(2)*L])

# problem constants:
DIMENSIONS = 10  # number of dimensions
BOUND_LOW, BOUND_UP = 64.516, 22580.6  # boundaries for all dimensions

# Genetic Algorithm constants:
POPULATION_SIZE = 60
P_CROSSOVER = 0.9  # probability for crossover (0.9,0.1), (0.3, 0.3)
P_MUTATION = 0.1   # (try also 0.5) probability for mutating an individual
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 10
CROWDING_FACTOR = 20.0  # crowding factor for crossover and mutation

# Define fintness trategy (min or max)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class as list, using fitness function as the definition above
creator.create("Individual", list, fitness=creator.FitnessMin)

# Function random float vector DIMENSIONS demensions with values in [low,up] 
def randomFloat(low, up):
	return [random.uniform(l, u) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

# Calculate weights
def calWeight(individual):
	total = 0
	for i in range(0, DIMENSIONS - 1):
		total+= P*LENGTHS[i]*individual[i]
	return total/1000000

# Calculate fitness function
def fitnessFunction(individual):
	v=0
	fx = calWeight(individual)
	for i in range(0, DIMENSIONS - 1):
		v+=1 if (S*FORCES[i]/individual[i] >= SIGMS[i]) else 0
	return fx*(1000*v + 1),


def eggholder(individual):
	x = individual[0]
	y = individual[1]
	f = (-(y + 47.0) * np.sin(np.sqrt(abs(x/2.0 + (y + 47.0)))) - x * np.sin(np.sqrt(abs(x - (y + 47.0)))))
	return f,  # return a tuple

# Init population and other parameters
def init():
	global POPULATION
	global SIGMS

	toolbox = getToolbox()
	random.seed(datetime.now())

	# create initial population (generation 0):
	POPULATION = toolbox.populationCreator(n=POPULATION_SIZE)

	# Init forces, sigmas
	#aOptimized = np.array([19722.54, 64.52, 14974.16, 9799.98, 64.52, 361.29, 4851.60, 13529.01, 13838.68, 64.52])
	#SIGMS = [round(S*f/a, 2) for f,a in zip(FORCES,aOptimized)]

	SIGMS = np.random.uniform(0, 1, DIMENSIONS)

	# print("Weight Optimized:", calWeight(aOptimized))
	print("SIGMAS:", SIGMS)

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

	toolbox.register("evaluate", fitnessFunction)

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
	random.seed(datetime.now())
	init()

	agaBest, agaMins, agaMeans = agaSolution()
	best, gaMins, gaMeans = gaSolution()

	print("-- Best Individual = ", best)
	print("-- Best Fitness = ", best.fitness.values[0])

	print("-- AGA Best Individual = ", agaBest)
	print("-- AGA Best Fitness = ", agaBest.fitness.values[0])

	# plot statistics
	sns.set_style("whitegrid")
	plt.plot(gaMins, color='red')
	plt.plot(agaMins, color='blue')
	plt.xlabel('Generation')
	plt.ylabel('F({X})')
	plt.gca().legend(('GA Mins','AGA Mins'))
	plt.show()

if __name__ == "__main__":
	main()