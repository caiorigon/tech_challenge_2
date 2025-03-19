import random
import math
import copy
from typing import List, Tuple
import numpy as np
from enum import Enum
from abc import ABC
from PIL import Image, ImageDraw


class TypeOfProblem(Enum):
    TSP = 1
    POLYGON = 2


class Problem(ABC):
    typeOfProblem: TypeOfProblem
    target: any


class Polygon:
    width: int
    height: int
    numPolygons: int
    numVertices: int
    color: Tuple[int, int, int, int]
    target_image: Image

    def __init__(
        self,
        width: int,
        height: int,
        numPolygons: int,
        numVertices: int,
        target_image: Image,
    ):
        self.width = width
        self.height = height
        self.numPolygons = numPolygons
        self.numVertices = numVertices
        self.target_image = target_image


class TSPProblem(Problem):
    def __init__(self, target: List[Tuple[float, float]]):
        self.typeOfProblem = TypeOfProblem.TSP
        self.target = target


class PolygonProblem(Problem):
    def __init__(self, target: Polygon):
        self.typeOfProblem = TypeOfProblem.POLYGON
        self.target = target


# individual is a possible solution
# each genes is the proposed solution
# fitness is the quality of the solution
class Individual:
    def __init__(self, problem: Problem, genes: any, fitness: float = 0.0):
        self.problem = problem
        self.genes = genes
        self.fitness = fitness
        self.calculate_fitness()

    def __repr__(self):
        return f"Individual with Genes: {self.genes}, Fitness: {self.fitness}\n"

    def calculate_fitness(self):

        match self.problem.typeOfProblem:
            case TypeOfProblem.TSP:
                n = len(self.genes)
                self.fitness = 0
                for i in range(n):
                    self.fitness += self.__calculate_euclidean_distance(
                        self.genes[i], self.genes[(i + 1) % n]
                    )
                self.fitness = round(self.fitness, 3)
            case TypeOfProblem.POLYGON:
                self.fitness = self.__polygon_fitness()

    # Calculate fitness
    def __polygon_fitness(self):
        # Draw the individual
        img = self.draw_individual()

        # Get pixel data
        img_data = np.array(img)
        target_data = np.array(self.problem.target.target_image)

        # Sample pixels (for performance)
        sample_rate = 10  # Sample every 10th pixel
        img_sampled = img_data[::sample_rate, ::sample_rate]
        target_sampled = target_data[::sample_rate, ::sample_rate]

        # Calculate difference (normalized Manhattan distance)
        diff = np.sum(
            np.abs(img_sampled.astype(np.int32) - target_sampled.astype(np.int32))
        )
        max_diff = 255 * 3 * img_sampled.shape[0] * img_sampled.shape[1]

        # Convert to fitness (0-100%, higher is better)
        return 100 * (1 - diff / max_diff)

    # Draw an individual
    def draw_individual(self):
        width = self.problem.target.width
        height = self.problem.target.height

        img = Image.new("RGBA", (width, height), color=(255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        for polygon in self.genes:
            draw.polygon(polygon["vertices"], fill=polygon["color"])

        # Convert to RGB for comparison with target
        return img.convert("RGB")

    def __calculate_euclidean_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def mutate(self, mutation_probability: float):
        match self.problem.typeOfProblem:
            case TypeOfProblem.TSP:
                self.__tsp_mutate(mutation_probability)
            case TypeOfProblem.POLYGON:
                self.__polygon_mutate(mutation_probability)
            case _:
                raise Exception("Invalid problem type")

    def __polygon_mutate(self, mutation_rate: float):
        for i in range(len(self.genes)):
            # Mutate polygon position
            if random.random() < mutation_rate:
                move_x = (random.random() - 0.5) * 20
                move_y = (random.random() - 0.5) * 20

                self.genes[i]["vertices"] = [
                    (x + move_x, y + move_y) for x, y in self.genes[i]["vertices"]
                ]

            # Mutate individual vertices
            for j in range(len(self.genes[i]["vertices"])):
                if random.random() < mutation_rate:
                    x, y = self.genes[i]["vertices"][j]
                    x += (random.random() - 0.5) * 10
                    y += (random.random() - 0.5) * 10
                    self.genes[i]["vertices"][j] = (x, y)

            # Mutate color
            if random.random() < mutation_rate:
                r, g, b, a = self.genes[i]["color"]
                r = max(0, min(255, r + int((random.random() - 0.5) * 50)))
                g = max(0, min(255, g + int((random.random() - 0.5) * 50)))
                b = max(0, min(255, b + int((random.random() - 0.5) * 50)))
                a = max(20, min(200, a + int((random.random() - 0.5) * 30)))
                self.genes[i]["color"] = (r, g, b, a)

    # TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two.
    def __tsp_mutate(self, mutation_probability: float):

        mutated_genes = copy.deepcopy(self.genes)

        # Check if mutation should occur
        if random.random() < mutation_probability:

            # Ensure there are at least two cities to perform a swap
            if len(self.genes) < 2:
                return self.genes

            # Select a random index (excluding the last index) for swapping
            index = random.randint(0, len(self.genes) - 2)

            # Swap the cities at the selected index and the next index
            mutated_genes[index], mutated_genes[index + 1] = (
                self.genes[index + 1],
                self.genes[index],
            )

        self.genes = mutated_genes


# population is a group of solutions
class GeneticAlgorithm:
    population: List[Individual] = []

    def __init__(self, problem: Problem, population_size: int):
        self.problem = problem
        self.population_size = population_size

        self.__create_random_population()

    def __create_random_population(self):
        match self.problem.typeOfProblem:
            case TypeOfProblem.TSP:
                random_population = [
                    self.__generate_random_coordinates(self.problem.target)
                    for _ in range(self.population_size)
                ]

                for individual in random_population:
                    self.population.append(Individual(self.problem, individual))

            case TypeOfProblem.POLYGON:
                random_population = [
                    self.__generate_random_polygon(
                        self.problem.target.width,
                        self.problem.target.height,
                        self.problem.target.numPolygons,
                        self.problem.target.numVertices,
                    )
                    for _ in range(self.population_size)
                ]
                for individual in random_population:
                    self.population.append(Individual(self.problem, individual))
            case _:
                raise Exception("Invalid problem type")

    # Function to create a random polygon to start the GA
    def __generate_random_polygon(self, width, height, num_polygons, num_vertices):
        list_polygons = []
        for _ in range(num_polygons):
            # Random center position
            center_x = random.randint(0, width)
            center_y = random.randint(0, height)

            # Random radius
            radius = random.randint(10, 50)

            # Create vertices around center
            vertices = []
            for i in range(num_vertices):
                angle = (i / num_vertices) * 2 * np.pi
                # Vary the radius slightly for each vertex
                vertex_radius = radius * (0.7 + random.random() * 0.6)

                x = center_x + vertex_radius * np.cos(angle)
                y = center_y + vertex_radius * np.sin(angle)
                vertices.append((x, y))

            # Random color with alpha
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(30, 150),  # Alpha (transparency)
            )
            list_polygons.append({"vertices": vertices, "color": color})

        return list_polygons

    def __generate_random_coordinates(
        self, coordinates: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        return random.sample(coordinates, len(coordinates))

    def sort_population(
        self,
    ) -> List[Individual]:
        match self.problem.typeOfProblem:
            case TypeOfProblem.TSP:
                sorted_population = sorted(self.population, key=lambda x: x.fitness)
            case TypeOfProblem.POLYGON:
                sorted_population = sorted(
                    self.population, key=lambda x: x.fitness, reverse=True
                )
            case _:
                sorted_population = self.population
        return sorted_population

    def calculate_population_fitness(self):
        for individual in self.population:
            individual.calculate_fitness()

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        match self.problem.typeOfProblem:
            case TypeOfProblem.TSP:
                return self.__tsp_crossover(parent1, parent2)
            case TypeOfProblem.POLYGON:
                return self.__polygon_crossover(parent1, parent2)
            case _:
                raise Exception("Invalid problem type")

    def __polygon_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Individual:
        child = Individual(parent1.problem, [])

        # TODO make a more complex crossover for polygons
        for i in range(len(parent1.genes)):
            # 50% chance of inheriting from each parent
            if random.random() < 0.5:
                child.genes.append(copy.deepcopy(parent1.genes[i]))
            else:
                child.genes.append(copy.deepcopy(parent2.genes[i]))

        return child

    def __tsp_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        length = len(parent1.genes)

        # Choose two random indices for the crossover
        start_index = random.randint(0, length - 1)
        end_index = random.randint(start_index + 1, length)

        # Initialize the child with a copy of the substring from parent1
        child = Individual(problem=parent1.problem, genes=[])
        child.genes = parent1.genes[start_index:end_index]

        # Fill in the remaining positions with genes from parent2
        remaining_positions = [
            i for i in range(length) if i < start_index or i >= end_index
        ]
        remaining_genes = [gene for gene in parent2.genes if gene not in child.genes]

        for position, gene in zip(remaining_positions, remaining_genes):
            child.genes.insert(position, gene)

        return child

    def tournament_selection(self, tournament_size=3):
        best_index = random.randint(0, len(self.population) - 1)

        for _ in range(tournament_size - 1):
            idx = random.randint(0, len(self.population) - 1)
            if self.population[idx].fitness > self.population[best_index].fitness:
                best_index = idx

        return copy.deepcopy(self.population[best_index])
