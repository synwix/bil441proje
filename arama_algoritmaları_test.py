import random
import time
import copy
import numpy as np

# Game of Life kurallarını uygulayan fonksiyon
def next_generation(grid):
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            neighbors = sum(grid[(i + di) % rows][(j + dj) % cols] 
                            for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0))
            if grid[i][j] == 1:
                new_grid[i][j] = 1 if neighbors in [2, 3] else 0
            else:
                new_grid[i][j] = 1 if neighbors == 3 else 0
    return new_grid

# 5 nesil ilerletme
def evolve_grid(grid, steps=5):
    current = copy.deepcopy(grid)
    for _ in range(steps):
        current = next_generation(current)
    return current

# Hamming mesafesi hesaplama
def hamming_distance(grid, target):
    return sum(1 for i in range(5) for j in range(5) if grid[i][j] != target[i][j])

# Rastgele 5x5 ızgara oluşturma
def random_grid():
    return [[random.randint(0, 1) for _ in range(5)] for _ in range(5)]

# Hedef glider
target_glider = [
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

########################## Genetik Algoritmalar
def crossover(parent1, parent2):
    split = random.randint(0, 4)
    child = parent1[:split] + parent2[split:]
    return [row[:] for row in child]

def mutate(grid, rate=0.1):
    new_grid = [row[:] for row in grid]
    for i in range(5):
        for j in range(5):
            if random.random() < rate:
                new_grid[i][j] = 1 - new_grid[i][j]
    return new_grid

def genetic_algorithm(target, pop_size=100, max_generations=100):
    population = [random_grid() for _ in range(pop_size)]
    start_time = time.time()
    
    for gen in range(max_generations):
        scores = []
        for individual in population:
            evolved = evolve_grid(individual)
            score = 25 - hamming_distance(evolved, target)
            scores.append((individual, score))
            if score == 25:
                end_time = time.time()
                return individual, end_time - start_time
        
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [scores[i][0] for i in range(int(pop_size * 0.2))] + \
                  random.sample(population, int(pop_size * 0.8))
        
        new_population = []
        while len(new_population) < pop_size:
            if random.random() < 0.8:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
            else:
                child = random.choice(selected)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

######################## Tepe Tırmanma Araması
def get_neighbors(grid):
    neighbors = []
    for i in range(5):
        for j in range(5):
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 1 - new_grid[i][j]
            neighbors.append(new_grid)
    return neighbors

def hill_climbing(target, max_iterations=100):
    current = random_grid()
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = 25 - hamming_distance(current_evolved, target)
        
        if current_score == 25:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(neighbor)
            score = 25 - hamming_distance(neighbor_evolved, target)
            if score > best_score:
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

######################## Benzetimli Tavlama Araması
import math

def simulated_annealing(target, max_iterations=1000, initial_temp=100):
    current = random_grid()
    temp = initial_temp
    start_time = time.time()
    
    for i in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = 25 - hamming_distance(current_evolved, target)
        
        if current_score == 25:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))
        neighbor_evolved = evolve_grid(neighbor)
        neighbor_score = 25 - hamming_distance(neighbor_evolved, target)
        
        if neighbor_score > current_score:
            current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < math.exp(delta / temp):
                current = neighbor
        
        temp *= 0.95
    
    end_time = time.time()
    return current, end_time - start_time

######################## Yerel Işın Araması

def local_beam_search(target, beam_width=10, max_iterations=100):
    beams = [random_grid() for _ in range(beam_width)]
    start_time = time.time()
    
    for _ in range(max_iterations):
        scores = []
        for beam in beams:
            beam_evolved = evolve_grid(beam)
            score = 25 - hamming_distance(beam_evolved, target)
            scores.append((beam, score))
            if score == 25:
                end_time = time.time()
                return beam, end_time - start_time
        
        candidates = []
        for beam in beams:
            neighbors = get_neighbors(beam)
            for neighbor in neighbors:
                neighbor_evolved = evolve_grid(neighbor)
                score = 25 - hamming_distance(neighbor_evolved, target)
                candidates.append((neighbor, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

######################## Karşılaştırma Kodları
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

# Algoritmaları çalıştır ve sürelerini karşılaştır
algorithms = {
    "Genetic Algorithm": genetic_algorithm,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Local Beam Search": local_beam_search
}

results = {}
for name, algo in algorithms.items():
    solution, duration = algo(target_glider)
    evolved = evolve_grid(solution)
    score = 25 - hamming_distance(evolved, target_glider)
    results[name] = (solution, duration, score)
    
    print(f"{name}:")
    print("Başlangıç Izgara:")
    print_grid(solution)
    print("5. Nesil:")
    print_grid(evolved)
    print(f"Süre: {duration:.4f} saniye")
    print(f"Puan: {score}")
    print("-" * 20)

# Süre karşılaştırması
print("Süre Karşılaştırması:")
for name, (solution, duration, score) in results.items():
    print(f"{name}: {duration:.4f} saniye (Puan: {score})")