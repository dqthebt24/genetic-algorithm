from deap import tools
import random
import numpy as np
from deap import algorithms

def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
	# Create a logbook
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	# Calculate fitness for all population
	fitnesses = map(toolbox.evaluate, population)

	# Set fitness to valid value
	for ind, fit in zip(population, fitnesses):
		ind.fitness.values = fit

	if halloffame is None:
		raise ValueError("halloffame parameter must not be empty!")

	# Get some best individual
	halloffame.update(population)
	hof_size = len(halloffame.items) if halloffame.items else 0

	record = stats.compile(population) if stats else {}
	logbook.record(gen=0, nevals=len(population), **record)
	
	# For logbook debuging
	if verbose:
		print(logbook.stream)

	for gen in range(1, ngen + 1):
		# Select the next generation individuals
		offspring = toolbox.select(population, len(population) - hof_size)

		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))

		'''
		Implement cross over and mutation
		'''
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):

			# cross two individuals with probability CXPB
			if random.random() < cxpb:
				toolbox.mate(child1, child2)

				# fitness values of the children
				# must be recalculated later
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			# mutate an individual with probability MUTPB
			if random.random() < mutpb:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		# Using DEAP's algorithm
		# offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# add the best back to population:
		offspring.extend(halloffame.items)

		# # Update the hall of fame with the generated individuals
		halloffame.update(offspring)

		# Replace the current population by the offspring
		population[:] = offspring

		# Append the current generation statistics to the logbook
		record = stats.compile(population) if stats else {}
		print("Records: {0}".format(record))
		logbook.record(gen=gen, nevals=len(invalid_ind), **record)
		if verbose:
			print(logbook.stream)

	return population, logbook

def eaAdaptiveWithElitism(population, toolbox, ngen, k1, k2, k3, k4, stats=None,
             halloffame=None, verbose=__debug__):
	def getMinAvg(record):
		return record['min'], record['avg']

	def calPc(fitness):
		# print("Fitness= {}; fMin={}; fAvg={}; k1 = {}".format(fitness, fMin, fAvg, k1))
		return (k1*(fMin - fitness)/(fMin - fAvg)) if fitness >= fAvg else k2;

	def calPm(fitness):
		return (k3*(fMin - fitness)/(fMin - fAvg)) if fitness >= fAvg else k4;

	# Create a logbook
	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

	# Calculate fitness for all population
	fitnesses = map(toolbox.evaluate, population)

	# Set fitness to valid value
	for ind, fit in zip(population, fitnesses):
		ind.fitness.values = fit

	if halloffame is None:
		raise ValueError("halloffame parameter must not be empty!")

	# Get some best individual
	halloffame.update(population)
	hof_size = len(halloffame.items) if halloffame.items else 0

	record = stats.compile(population) if stats else {}
	logbook.record(gen=0, nevals=len(population), **record)
	
	# Get min, avg
	fMin, fAvg = getMinAvg(record)

	# For logbook debuging
	if verbose:
		print(logbook.stream)

	for gen in range(1, ngen + 1):
		# Select the next generation individuals
		offspring = toolbox.select(population, len(population) - hof_size)

		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))

		'''
		Implement cross over and mutation
		'''
		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			cxpb = calPc(np.minimum(child1.fitness.values[0], child2.fitness.values[0]))
			# print("Pc = ", cxpb)
			# cross two individuals with probability CXPB
			if random.random() < cxpb:
				toolbox.mate(child1, child2)

				# fitness values of the children
				# must be recalculated later
				del child1.fitness.values
				del child2.fitness.values

		# Recalculate fitness after cross over
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		for mutant in offspring:
			mutpb = calPm(mutant.fitness.values[0])
			# mutate an individual with probability MUTPB
			if random.random() < mutpb:
				toolbox.mutate(mutant)
				del mutant.fitness.values

		# # Vary the pool of individuals
		# offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		# add the best back to population:
		offspring.extend(halloffame.items)

		# # Update the hall of fame with the generated individuals
		halloffame.update(offspring)

		# Replace the current population by the offspring
		population[:] = offspring

		# Append the current generation statistics to the logbook
		record = stats.compile(population) if stats else {}
		print("Records: {0}".format(record))
		
		print("Min: {}; Avg{}".format(fMin, fAvg))
		logbook.record(gen=gen, nevals=len(invalid_ind), **record)
		if verbose:
			print(logbook.stream)

	return population, logbook