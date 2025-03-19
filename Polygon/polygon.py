import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import copy
from genetic_algorithm import (
    GeneticAlgorithm,
    PolygonProblem,
    Polygon,
)

# Set page configuration
st.set_page_config(page_title="Image Recreation with Polygons", layout="wide")

# Title
st.title("Image Recreation with Polygons using Genetic Algorithm")

# Sidebar title
st.sidebar.header("Genetic Algorithm Parameters")

# genetic algorithm parameters
population_size = st.sidebar.slider("Population Size", 10, 100, 30)
num_polygons = st.sidebar.slider("Number of Polygons", 10, 200, 50)
num_vertices = st.sidebar.slider("Vertices per Polygon", 3, 10, 6)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.3, 0.05)
elite_count = st.sidebar.slider("Elite Count", 1, 5, 2)
tournament_size = st.sidebar.slider("Tournament Size", 2, 10, 3)

# Selection of the target image
st.sidebar.header("Target Image")
target_image_option = st.sidebar.selectbox(
    "Choose a target image", ["Simple Shapes", "Upload Your Own"]
)

# Initialize session state to store GA state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.generation = 0
    st.session_state.GA = None
    st.session_state.best_fitnessGA = 0
    st.session_state.running = False
    st.session_state.target_image = None
    st.session_state.image_width = 400
    st.session_state.image_height = 300
    st.session_state.fitness_history = []


# Function to create a target image with simple shapes
def create_simple_target():
    width = st.session_state.image_width
    height = st.session_state.image_height

    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw a blue-green gradient background
    for y in range(height):
        r = int(0)
        g = int(y / height * 200)
        b = int(200 - y / height * 150)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Draw a red circle
    draw.ellipse(
        [(width // 4, height // 4), (width * 3 // 4, height * 3 // 4)],
        fill=(200, 50, 50, 128),
    )

    # Draw a yellow rectangle
    draw.rectangle(
        [(width // 6, height // 6), (width // 3, height // 3)], fill=(240, 240, 30)
    )

    return img


# Handle target image selection
if target_image_option == "Simple Shapes":
    if st.session_state.target_image is None or st.sidebar.button(
        "Generate New Simple Target"
    ):
        st.session_state.target_image = create_simple_target()
        st.session_state.initialized = False  # Reset GA when target changes
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            img = img.resize(
                (st.session_state.image_width, st.session_state.image_height)
            )
            img = img.convert("RGB")
            st.session_state.target_image = img
            st.session_state.initialized = False  # Reset GA when target changes
        except Exception as e:
            st.error(f"Error loading image: {e}")


# Initialize population
def initialize_population():

    polygon = Polygon(
        width=st.session_state.image_width,
        height=st.session_state.image_height,
        numPolygons=num_polygons,
        numVertices=num_vertices,
        target_image=st.session_state.target_image,
    )

    ga = GeneticAlgorithm(PolygonProblem(polygon), population_size)
    st.session_state.GA = ga

    # Sort by fitness (descending)
    st.session_state.GA.population = ga.sort_population()

    if len(st.session_state.GA.population) > 0:
        st.session_state.best_fitnessGA = st.session_state.GA.population[0].fitness

    st.session_state.generation = 0
    st.session_state.fitness_history = [st.session_state.best_fitnessGA]
    st.session_state.initialized = True


# Evolution step
def evolution_step():
    if not st.session_state.running or not st.session_state.initialized:
        return

    new_populationGA = []

    # Elitism - keep the best individuals
    for i in range(min(elite_count, len(st.session_state.GA.population))):
        new_populationGA.append(copy.deepcopy(st.session_state.GA.population[i]))

    # Create the rest of the population
    while len(new_populationGA) < population_size:
        parent1GA = st.session_state.GA.tournament_selection(tournament_size)
        parent2GA = st.session_state.GA.tournament_selection(tournament_size)

        childGA = st.session_state.GA.crossover(parent1GA, parent2GA)
        childGA.mutate(mutation_rate)

        new_populationGA.append(childGA)

    # Evaluate new population
    st.session_state.GA.population = new_populationGA
    st.session_state.GA.calculate_population_fitness()
    st.session_state.GA.population = st.session_state.GA.sort_population()
    st.session_state.best_fitnessGA = st.session_state.GA.population[0].fitness

    st.session_state.generation += 1
    st.session_state.fitness_history.append(st.session_state.best_fitnessGA)


# Setup the UI layout
col1, col2 = st.columns(2)

# Display target image
with col1:
    st.header("Target Image")
    if st.session_state.target_image:
        st.image(st.session_state.target_image, use_container_width=True)
    else:
        st.info("Please select or generate a target image from the sidebar.")

# Display current best solution
with col2:
    st.header(f"Best Solution (Generation {st.session_state.generation})")
    if st.session_state.initialized and len(st.session_state.GA.population) > 0:
        best_image = st.session_state.GA.population[0].draw_individual()
        st.image(best_image, use_container_width=True)
    else:
        st.info("Evolution not started yet.")

# Control buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.session_state.target_image is not None:
        if not st.session_state.initialized:
            if st.button("Initialize Population"):
                initialize_population()
        else:
            start_stop = st.button("Start" if not st.session_state.running else "Stop")
            if start_stop:
                st.session_state.running = not st.session_state.running

with col_btn2:
    if st.session_state.initialized:
        if st.button("Reset"):
            st.session_state.running = False
            initialize_population()

with col_btn3:
    if st.session_state.initialized and len(st.session_state.GA.population) > 0:
        if st.button("Single Step"):
            evolution_step()

# Display evolution metrics
if st.session_state.initialized:
    st.header("Evolution Progress")

    # Display metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Generation", st.session_state.generation)
    with metrics_col2:
        st.metric("Best Fitness", f"{st.session_state.best_fitnessGA:.2f}%")
    with metrics_col3:
        st.metric("Population Size", population_size)

    # Display fitness history chart
    if len(st.session_state.fitness_history) > 1:
        st.subheader("Fitness History")
        fig, ax = plt.subplots()
        ax.plot(st.session_state.fitness_history)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (%)")
        ax.set_ylim(0, 100)
        ax.grid(True)
        st.pyplot(fig)

# Main loop for evolution
if st.session_state.running and st.session_state.initialized:
    evolution_step()
    st.rerun()
