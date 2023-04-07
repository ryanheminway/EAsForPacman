# -*- coding: utf-8 -*-
"""
geneticTraining.py contains the implmentation for the Genetic Algorithm
to train Pacman agents to play based on a policy function implemented as a 
neural network. 
"""
import numpy as np
from util import *
from dill_utils import map_with_dill
import pacmanNN

class GeneticAlgorithm:
    """
    Implements a generic genetic algorithm. Since specific versions of GA 
    require differences in selection and replication, this class should be
    subclassed.
    
    Inputs:
        fitness_fn: Function which takes an indivual as input and produces a 
            scalar value fitness score.
        num_genes: Number of genes attributed to each individual. Each gene is 
            single floating point value.
        num_individuals: Number of individuals to create in population.
        num_generations: Number of generations to run for.
        rate_mutation: Probability (0-1) to use mutation operator to produce 
            offspring.
        rate_crossover: Probability (0-1) to use crossover operator to produce
            offspring.
    """
    def __init__(self, fitness_fn, num_genes=40, num_individuals=100, num_generations=100,
                 rate_mutation=0.1, rate_crossover=1.0):
        assert(num_genes > 0)
        assert(num_individuals > 0)
        assert(num_generations > 0)
        assert(rate_mutation <= 1.0 and rate_mutation >= 0.0)
        assert(rate_crossover <= 1.0 and rate_crossover >= 0.0)
        self.num_genes = num_genes
        # Generate initial random pop
        self.individuals = np.array([GeneticAlgorithm.generate_random_individual(num_genes) for i in range(num_individuals)])
        self.num_generations = num_generations
        self.rate_mutation = rate_mutation
        self.rate_crossover = rate_crossover
        self.fitness_fn = fitness_fn
        self.num_individuals = num_individuals
        
        # Track generation
        self.generation = 0
        # Track best individual for each generation (index is generation)
        self.best_by_gen = np.empty((num_generations, num_genes))
    
    @staticmethod
    def generate_random_individual(num_genes):
        """
        Generate a random individual chromosome. Resulting shape is 1xnum_genes
        """
        return pacmanNN.PacmanControllerModel().initFlatWeights()
    
    @staticmethod
    def generate_random_allele():
        return pacmanNN.PacmanControllerModel().initSingleWeight()
        
    def run(self):
        """
        Run the full GA, for the specified number of generations.

        Return:
            The an array of the best individual from each generation
        """
        for i in range(self.num_generations):
            self.step()
            
        # Return the best individual for each gen
        return self.best_by_gen 
    
    def step(self):
        """
        Iterate the GA by one generation. Steps based on current state of 
        algorithm. Implementation of a single generation should be defined
        by subclass.

        Return:
            None.
        """
        raiseNotDefined()
        
    def crossover(self, a, b):
        """
        Defines how to perform crossover between two individuals, a and b. 
        Crossover can be implemented in single-point, multi-point, or uniform
        style and thus is up to the implementation.
        
        Return:
            Child chromosome resulting from crossover between a and b.
        """
        raiseNotDefined()
        
    def mutate(self, a):
        """
        Defines how to perform mutation on a given individual a. Mutation
        should occur according to rate_mutation as a probability.
        
        Return:
            Individual corresponding to a mutated version of a.
        """
        raiseNotDefined()
        
    @staticmethod
    def singlePointCrossover(a, b, idx):
        """
        Combine parents a and b into a two children using single-point crossover.
        All chromosome information up to point idx in a is combined with all 
        information after point idx in b to create the first child. The second
        child is the inverse

        Return:
            Two individuals representing result of single-point crossover.
        """
        return (np.append(a[:idx], b[idx:]), np.append(b[:idx], a[idx:]))
    
    @staticmethod
    def tournamentSelect(scores, population, tournySize=5):
        winnerScore = -1000000
        winnerIdx = -1
        for i in range(tournySize):
            # random generate a contestant
            contestantIdx = np.random.randint(0, scores.size)
            # They are current winner if their fitness is highest
            if scores[contestantIdx] > winnerScore:
                winnerScore = scores[contestantIdx]
                winnerIdx = contestantIdx
                
        return population[winnerIdx]
    
    @staticmethod 
    def rouletteSelect(scores, population):
        minScore = abs(min(scores))
        newScores = [(s + minScore) for s in scores]
        totalScore = sum(newScores)
        selection_probs = [score/totalScore for score in newScores]
        # Roulette wheel where size of wedge is proportial to fitness
        return population[np.random.choice(len(population), p=selection_probs)]
        
        
                
class SteadyStateGA(GeneticAlgorithm):
    """
    Iterative or "Steady State" version of GA. At each generation, two parents
    are selected for reproduction. Their offspring, created via a combination 
    of crossover and/or mutation, is reinserted into the population. The 
    offspring replaces the worst 2 members of the current population.
    """
    def __init__(self, fitness_fn, num_genes=4, num_individuals=100, num_generations=20000,
                 rate_mutation=0.1, rate_crossover=1.0, tourny_size=10):
        super().__init__(fitness_fn, num_genes, num_individuals, num_generations, 
                       rate_mutation, rate_crossover)
        #self.scores = np.asarray([self.fitness_fn(x) for x in self.individuals])

        self.scores = np.asarray(map_with_dill(self.fitness_fn, self.individuals))
        self.tournySize = tournySize
        
    def step(self):
        """
        Single generational step of Steady State GA. Executes the following 
        routine:
            1. Pick 2 parents using tournament selection with a tournament
               size of 2. The best individual is chosen from each tournament.
            2. Perform single point crossover between parents to generate two children.
            3. Based on rate_mutation, randomly mutate generated children.
            4. Replace current worst individuals in population with children if
               the children have better fitness.
        """
        #print("Running GA generation #", self.generation)
        # preallocate parents. This will get overwritten
        parents = np.array([self.individuals[0], self.individuals[1]])
        
        # 1. Pick 2 parents using tournament selection
        for i in range(2):
            # (NOTE Ryan) Using tourny size of 5. Have used size of 2 before
            parents[i] = GeneticAlgorithm.tournamentSelect(self.scores, self.individuals, tournySize=self.tournySize)
                
        # 2. Perform single point crossover between parents to generate child.
        (childOne, childTwo) = self.crossover(parents[0], parents[1], self.rate_crossover)
        
        # 3. Based on rate_mutation, randomly mutate generated child.
        childOne = self.mutate(childOne)
        childTwo = self.mutate(childTwo)
        
        # 4. Replace current worst two individuals in population with children.
        worstIdxs = np.argpartition(self.scores, 2)[:2]
        childOneScore = self.fitness_fn(childOne)
        childTwoScore = self.fitness_fn(childTwo)
        if (childOneScore > self.scores[worstIdxs[0]]):
            # ChildOne is better than worst
            self.individuals[worstIdxs[0]] = childOne
            self.scores[worstIdxs[0]] = childOneScore
        if (childTwoScore > self.scores[worstIdxs[1]]):
            # ChildTwo is better than worst
            self.individuals[worstIdxs[1]] = childTwo
            self.scores[worstIdxs[1]] = childTwoScore

        # Update tracking
        best_idx = np.argmax(self.scores)
        self.best_by_gen[self.generation] = self.individuals[best_idx]
        self.generation += 1
        if (self.generation == (self.num_generations - 1)):
            print("Best fitness value of this generation: ", self.scores[best_idx])
        
    def crossover(self, a, b, prob_crossover):
        if np.random.random() < prob_crossover:      
            idx = np.random.randint(0,a.size)
            return GeneticAlgorithm.singlePointCrossover(a, b, idx)
        else:
            # If crossover didn't occur just return parent a
            return (a.copy(), b.copy())
    
    def mutate(self, a):
        a_copy = a.copy()
        # Each allele in the chromosome has a chance to be mutated
        for i in range(a_copy.size):
            mutate_chance = np.random.random()
            if mutate_chance < self.rate_mutation:
                a_copy[i] = self.generate_random_allele()     
        return a_copy
    
class GenerationalGA(GeneticAlgorithm):
    """
    "Generational" version of GA. At each generation, an entirely new population
    is created (with the exception of "elites"). Two parents are selected, and 
    two offspring are produced by those parents. The offspring are placed into
    the next generation's population. This process repeats until the next 
    generation's population has sufficient size to continue. "Elites" are 
    individuals with the highest fitness of a given generation. These individuals
    survive between generations. 
    """
    def __init__(self, fitness_fn, num_genes=4, num_individuals=100, num_generations=1000,
                 rate_mutation=0.1, rate_crossover=0.5, proportion_elite=0.1, tourny_size=10, selection_type="roulette"):
        super().__init__(fitness_fn, num_genes, num_individuals, num_generations, 
                       rate_mutation, rate_crossover)
        assert(proportion_elite <= 1 and proportion_elite >= 0)
        self.selection_type = selection_type
        self.proportion_elite = proportion_elite
        self.num_elites = int(proportion_elite * num_individuals)
        self.tourny_size = tourny_size
        self.scores = np.asarray(map_with_dill(self.fitness_fn, self.individuals))
        
    def step(self):
        """
        Single generation of Generational GA. Executes the following 
        routine:
            1. Copy elites to next generation's population
            2. Sample two parents from current population using tournament
               selection. 
            3. Use these parents to generate two offspring.
            4. Add offspring to next generation's population
            5. Repeat steps 3-5 until next generation has large enough population.
        """
        #print("Running GA generation #", self.generation)
        # preallocate parents. This will get overwritten
        parents = np.array([self.individuals[0], self.individuals[1]])
        
        # 1. Copy Elites
        nextPopulation = self.individuals.copy()
        nextScores = self.scores.copy()
        bestIndices = np.argpartition(self.scores, -self.num_elites)[-self.num_elites:]
        for i in range(self.num_elites):
            nextPopulation[i] = self.individuals[bestIndices[i]]
            nextScores[i] = self.scores[bestIndices[i]]
        
        nextIdx = self.num_elites
        for n in range(int((self.num_individuals - self.num_elites) / 2)):
            # 2. Pick 2 parents using selection operator specified
            for i in range(2):
                if self.selection_type == "tournament":
                    parents[i] = GeneticAlgorithm.tournamentSelect(self.scores, self.individuals, tournySize=self.tourny_size)
                elif self.selection_type == "roulette":
                    parents[i] = GeneticAlgorithm.rouletteSelect(self.scores, self.individuals)
                else: # Truncated selection
                    # Use a random elite as parent
                    parents[i] = nextPopulation[np.random.randint(0, self.num_elites)]
                
            # Perform single point crossover between parents to generate child.
            (childOne, childTwo) = self.crossover(parents[0], parents[1], self.rate_crossover)
            
            # Based on rate_mutation, randomly mutate generated child.
            childOne = self.mutate(childOne)
            childTwo = self.mutate(childTwo)
            
            nextPopulation[nextIdx] = childOne
            nextIdx += 1
            nextPopulation[nextIdx] = childTwo
            nextIdx += 1
        
        # Parallel compute fitness for all new offspring
        tempNextScores = map_with_dill(self.fitness_fn, nextPopulation[self.num_elites:])
        nextIdx = self.num_elites
        for i in range(len(tempNextScores)):
            nextScores[nextIdx] = tempNextScores[i]
            nextIdx += 1
            
        
        # Update tracking
        self.scores = nextScores
        best_idx = np.argmax(self.scores)
        print("Best fitness value of this generation: ", self.scores[best_idx])
        self.individuals = nextPopulation
        self.best_by_gen[self.generation] = self.individuals[best_idx]
        self.generation += 1            
        
        
    def crossover(self, a, b, prob_crossover):
        if np.random.random() < prob_crossover:      
            idx = np.random.randint(0, a.size)
            return GeneticAlgorithm.singlePointCrossover(a, b, idx)
        else:
            # If crossover didn't occur just return parents
            return (a.copy(), b.copy())
    
    def mutate(self, a):
        a_copy = a.copy()
        # Each gene in the chromosome has a chance to be mutate
        for i in range(a_copy.size):
            mutate_chance = np.random.random()
            if mutate_chance < self.rate_mutation:
                a_copy[i] = self.generate_random_allele()            
        return a_copy
    
    
# -------------- TEST BED ----------------- #        
if __name__ == '__main__':
    """
    To test the GA is implemented appropriately, I am providing a simpler example
    where the GA should fit parameters of the following equation:
        y = w_1*x_1 + w_2*x_2 + w_3*x_3 + w_4*x_4
    
    With the specific equation we would like to solve:
        44 = w_1*(4) + w_2*(-2) + w_3*(3.5) + w_4*(5)
    """
    
    function_inputs = [4,-2,3.5,5]
    desired_output = 44

    def fitness_func(solution):
        output = np.sum(solution*function_inputs)
        fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)

        return fitness
    
    test_ga = GenerationalGA(fitness_func, num_genes=4, num_individuals=100, 
                            num_generations=100)
    
    best_by_gen = test_ga.run()
    for i in range(100):
        print("Solution for gen #", i)
        print(best_by_gen[i])
        print("Output produced: ", np.sum(best_by_gen[i]*function_inputs))
    