import random
import time
import copy

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

def check_glider_region(region, glider):
    diff = 0
    for i in range(3):
        for j in range(3):
            if region[i][j] != glider[i][j]:
                diff += 1
    return diff

def find_glider(grid, glider):
    rows, cols = len(grid), len(grid[0])
    min_diff = float('inf')
    
    for i in range(rows - 2):
        for j in range(cols - 2):
            region = [row[j:j+3] for row in grid[i:i+3]]
            diff = check_glider_region(region, glider)
            min_diff = min(min_diff, diff)
            if min_diff == 0:
                return 9
    
    return 9 - min_diff

def random_grid(rows, cols):
    return [[random.randint(0, 1) for _ in range(cols)] for _ in range(rows)]

def crossover(parent1, parent2):
    split = random.randint(0, len(parent1) - 1)
    child = parent1[:split] + parent2[split:]
    return [row[:] for row in child]

def mutate(grid, rate=0.1):
    new_grid = [row[:] for row in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if random.random() < rate:
                new_grid[i][j] = 1 - new_grid[i][j]
    return new_grid

def get_neighbors(grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 1 - new_grid[i][j]
            neighbors.append(new_grid)
    return neighbors

# Algoritmalar
def genetic_algorithm(target, rows=5, cols=5, pop_size=200, max_generations=500):
    population = [random_grid(rows, cols) for _ in range(pop_size)]
    start_time = time.time()
    
    for _ in range(max_generations):
        scores = []
        for individual in population:
            evolved = evolve_grid(individual)
            score = find_glider(evolved, target)
            scores.append((individual, score))
            if score == 9:
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
            child = mutate(child, rate=0.05)
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

def hill_climbing(target, rows=5, cols=5, max_iterations=100):
    current = random_grid(rows, cols)
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_glider(current_evolved, target)
        
        if current_score == 9:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(neighbor)
            score = find_glider(neighbor_evolved, target)
            if score > best_score:
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

import math
def simulated_annealing(target, rows=5, cols=5, max_iterations=1000, initial_temp=100):
    current = random_grid(rows, cols)
    temp = initial_temp
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_glider(current_evolved, target)
        
        if current_score == 9:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))
        neighbor_evolved = evolve_grid(neighbor)
        neighbor_score = find_glider(neighbor_evolved, target)
        
        if neighbor_score > current_score:
            current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < math.exp(delta / temp):
                current = neighbor
        
        temp *= 0.95
    
    end_time = time.time()
    return current, end_time - start_time

def local_beam_search(target, rows=5, cols=5, beam_width=10, max_iterations=100):
    beams = [random_grid(rows, cols) for _ in range(beam_width)]
    start_time = time.time()
    
    for _ in range(max_iterations):
        scores = []
        for beam in beams:
            beam_evolved = evolve_grid(beam)
            score = find_glider(beam_evolved, target)
            scores.append((beam, score))
            if score == 9:
                end_time = time.time()
                return beam, end_time - start_time
        
        candidates = []
        for beam in beams:
            neighbors = get_neighbors(beam)
            for neighbor in neighbors:
                neighbor_evolved = evolve_grid(neighbor)
                score = find_glider(neighbor_evolved, target)
                candidates.append((neighbor, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

# Hedef glider (3x3)
target_glider_3x3 = [
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
]

# Test
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

algorithms = {
    "Genetic Algorithm": genetic_algorithm,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Local Beam Search": local_beam_search
}

results = {}
for name, algo in algorithms.items():
    for size in [(5, 5), (10, 10)]:
        rows, cols = size
        solution, duration = algo(target_glider_3x3, rows=rows, cols=cols)
        evolved = evolve_grid(solution)
        score = find_glider(evolved, target_glider_3x3)
        results[f"{name} ({rows}x{cols})"] = (solution, duration, score)
        
        print(f"{name} ({rows}x{cols}):")
        print("Başlangıç Izgara:")
        print_grid(solution)
        print("5. Nesil:")
        print_grid(evolved)
        print(f"Süre: {duration:.4f} saniye")
        print(f"Puan: {score}")
        print("-" * 20)

print("Süre Karşılaştırması:")
for name, (solution, duration, score) in results.items():
    print(f"{name}: {duration:.4f} saniye (Puan: {score})")