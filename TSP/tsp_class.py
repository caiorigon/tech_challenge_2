# solving traveling salesman problem using genetic algorithm
import random
import itertools
import sys
import pygame
import numpy as np
from draw_functions import (
    draw_cities,
    draw_paths,
    draw_plot,
    draw_text,
)

sys.path.insert(1, "./")
from genetic_algorithm import GeneticAlgorithm, TSPProblem

# Define constants
# Pygame
WIDTH, HEIGHT = 800, 600
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

# Genetic Algorithm
N_CITIES = 15
POPULATION_SIZE = 100
MAX_GENERATIONS = 10000
MUTATION_PROBABILITY = 0.5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize problem
# using Random cities generation
cities_locations = [
    (
        random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS),  # eixo X
        random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS),  # eixo y
    )
    for _ in range(N_CITIES)
]

# initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traveling Salesman Problem")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)

# create initial population (each individual is a possible solution)
# TODO: can use heuristic(hotstart) to generate initial population

problem = TSPProblem(cities_locations)
geneticAlgorithm = GeneticAlgorithm(problem, POPULATION_SIZE)

best_fitness_values = []
old_best_fitness = 0
should_stop_count = 0
# Main Game Loop
running = True
paused = False

# Create options to close simulation
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE or event.key == pygame.K_BACKSPACE:
                running = False
            if event.key == pygame.K_PAUSE or event.key == pygame.K_p:
                paused = not paused
                print(f"Paused: {paused}")
                should_stop_count = 0

    if should_stop_count >= 200:
        paused = True

    if not paused:
        generation = next(generation_counter)
        if generation >= MAX_GENERATIONS:
            running = False

        screen.fill(WHITE)

        geneticAlgorithm.calculate_population_fitness()

        population_fitness = [
            individual.fitness for individual in geneticAlgorithm.population
        ]

        sorted_population = geneticAlgorithm.sort_population()

        best_individual = sorted_population[0]

        best_fitness = best_individual.fitness
        best_fitness_values.append(best_fitness)

        if old_best_fitness == best_fitness:
            should_stop_count += 1
        else:
            should_stop_count = 0

        draw_text(
            screen=screen, text=f"Generation: {generation}", coordinates=(50, 500)
        )
        draw_text(
            screen=screen, text=f"Best fitness: {best_fitness}", coordinates=(50, 550)
        )
        draw_plot(
            screen,
            x=list(range(len(best_fitness_values))),
            y=best_fitness_values,
            y_label="Fitness - Distance(pixels)",
        )

        draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        print(f"Best solution: {best_individual}")
        draw_paths(screen, best_individual.genes, BLUE, width=3)
        draw_paths(screen, geneticAlgorithm.population[1].genes, GRAY, width=1)

        new_population = [
            best_individual
        ]  # Keep the best individual, using elitisim mode

        while len(new_population) < POPULATION_SIZE:
            # # Selection
            # # simple solution based on first 10 solutions
            parent1, parent2 = random.choices(sorted_population[:10], k=2)

            # solution based on fitness probability
            probability = 1 / np.array(population_fitness)
            parent1, parent2 = random.choices(
                geneticAlgorithm.population, weights=probability, k=2
            )

            # mutation and crossover of childs
            child = geneticAlgorithm.crossover(parent1, parent2)
            child.mutate(MUTATION_PROBABILITY)
            new_population.append(child)

        old_best_fitness = best_fitness
        geneticAlgorithm.population = new_population

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
