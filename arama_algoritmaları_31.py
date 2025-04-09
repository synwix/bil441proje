import random
import time
import copy
import math
import statistics

# Game of Life kurallarını uygulayan fonksiyon (açık sınırlar - toroidal)
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

# 5 nesil ilerletme fonksiyonu
def evolve_grid(grid, steps=5):
    current = copy.deepcopy(grid)
    for _ in range(steps):
        current = next_generation(current)
    return current

# 3x3 bölgenin belirli bir desenle farkını hesapla
def check_region(region, pattern):
    diff = 0
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if region[i][j] != pattern[i][j]:
                diff += 1
    return diff

# Izgarada yalnızca belirli bir deseni ara (tam eşleşme zorunlu, rotasyon yok)
def find_pattern(grid, pattern):
    rows, cols = len(grid), len(grid[0])
    pattern_rows, pattern_cols = len(pattern), len(pattern[0])
    max_score = pattern_rows * pattern_cols
    
    for i in range(rows - pattern_rows + 1):
        for j in range(cols - pattern_cols + 1):
            region = [row[j:j+pattern_cols] for row in grid[i:i+pattern_rows]]
            if all(region[r][c] == pattern[r][c] for r in range(pattern_rows) 
                   for c in range(pattern_cols)):
                return max_score
    
    min_diff = float('inf')
    for i in range(rows - pattern_rows + 1):
        for j in range(cols - pattern_cols + 1):
            region = [row[j:j+pattern_cols] for row in grid[i:i+pattern_rows]]
            diff = check_region(region, pattern)
            min_diff = min(min_diff, diff)
    return max_score - min_diff

# Rastgele ızgara oluşturma
def random_grid(rows, cols):
    return [[random.randint(0, 1) for _ in range(cols)] for _ in range(rows)]

# Çaprazlama (crossover) fonksiyonu
def crossover(parent1, parent2):
    split = random.randint(0, len(parent1) - 1)
    child = parent1[:split] + parent2[split:]
    return [row[:] for row in child]

# Mutasyon fonksiyonu
def mutate(grid, rate=0.1):
    new_grid = [row[:] for row in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if random.random() < rate:
                new_grid[i][j] = 1 - new_grid[i][j]
    return new_grid

# Komşu durumlar üretme
def get_neighbors(grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 1 - new_grid[i][j]
            neighbors.append(new_grid)
    return neighbors

# Genetik Algoritmalar
def genetic_algorithm(target_pattern, rows=5, cols=5, pop_size=200, max_generations=500):
    population = [random_grid(rows, cols) for _ in range(pop_size)]
    start_time = time.time()
    
    for gen in range(max_generations):
        scores = []
        for individual in population:
            evolved = evolve_grid(individual)
            score = find_pattern(evolved, target_pattern)
            scores.append((individual, score))
            max_score = len(target_pattern) * len(target_pattern[0])
            if score == max_score:
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

# Tepe Tırmanma Araması
def hill_climbing(target_pattern, rows=5, cols=5, max_iterations=100):
    current = random_grid(rows, cols)
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_pattern(current_evolved, target_pattern)
        
        max_score = len(target_pattern) * len(target_pattern[0])
        if current_score == max_score:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(neighbor)
            score = find_pattern(neighbor_evolved, target_pattern)
            if score > best_score:
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

# Benzetimli Tavlama
def simulated_annealing(target_pattern, rows=5, cols=5, max_iterations=1000, initial_temp=100):
    current = random_grid(rows, cols)
    temp = initial_temp
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_pattern(current_evolved, target_pattern)
        
        max_score = len(target_pattern) * len(target_pattern[0])
        if current_score == max_score:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))
        neighbor_evolved = evolve_grid(neighbor)
        neighbor_score = find_pattern(neighbor_evolved, target_pattern)
        
        if neighbor_score > current_score:
            current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < math.exp(delta / temp):
                current = neighbor
        
        temp *= 0.95
    
    end_time = time.time()
    return current, end_time - start_time

# Yerel Işın Araması
def local_beam_search(target_pattern, rows=5, cols=5, beam_width=10, max_iterations=100):
    beams = [random_grid(rows, cols) for _ in range(beam_width)]
    start_time = time.time()
    
    for _ in range(max_iterations):
        scores = []
        for beam in beams:
            beam_evolved = evolve_grid(beam)
            score = find_pattern(beam_evolved, target_pattern)
            scores.append((beam, score))
            max_score = len(target_pattern) * len(target_pattern[0])
            if score == max_score:
                end_time = time.time()
                return beam, end_time - start_time
        
        candidates = []
        for beam in beams:
            neighbors = get_neighbors(beam)
            for neighbor in neighbors:
                neighbor_evolved = evolve_grid(neighbor)
                score = find_pattern(neighbor_evolved, target_pattern)
                candidates.append((neighbor, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

# Hedef desen (Glider - Spaceship, tek yön)
target_pattern = [
    [0, 1, 0], 
    [0, 0, 1], 
    [1, 1, 1]
]

# Izgarayı yazdırma fonksiyonu
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

# Tek deneme testi
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
        solution, duration = algo(target_pattern, rows=rows, cols=cols)
        evolved = evolve_grid(solution)
        score = find_pattern(evolved, target_pattern)
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

# 100 deneme testi (ek istatistiklerle: medyan, min, max)
NUM_TRIALS = 100
trial_results = {
    name: {
        f"{rows}x{cols}": {"times": [], "success": 0, "failure": 0}
        for rows, cols in [(5, 5), (10, 10)]
    }
    for name in algorithms
}

for name, algo in algorithms.items():
    for size in [(5, 5), (10, 10)]:
        rows, cols = size
        key = f"{rows}x{cols}"
        for _ in range(NUM_TRIALS):
            solution, duration = algo(target_pattern, rows=rows, cols=cols)
            evolved = evolve_grid(solution)
            score = find_pattern(evolved, target_pattern)
            max_score = len(target_pattern) * len(target_pattern[0])
            if score == max_score:
                trial_results[name][key]["times"].append(duration)
                trial_results[name][key]["success"] += 1
            else:
                trial_results[name][key]["failure"] += 1

# Sonuçları raporla (ek istatistiklerle)
print("\n100 Deneme İstatistikleri:")
for name in algorithms:
    for size in [(5, 5), (10, 10)]:
        rows, cols = size
        key = f"{rows}x{cols}"
        times = trial_results[name][key]["times"]
        success = trial_results[name][key]["success"]
        failure = trial_results[name][key]["failure"]
        avg_time = sum(times) / len(times) if times else 0
        variance = sum((t - avg_time) ** 2 for t in times) / len(times) if times else 0
        std_dev = math.sqrt(variance) if variance > 0 else 0
        median_time = statistics.median(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        print(f"{name} ({key}):")
        print(f"  Bulma Sayısı: {success}")
        print(f"  Bulamama Sayısı: {failure}")
        print(f"  Ortalama Bulma Süresi (sadece başarılı denemeler): {avg_time:.4f} saniye")
        print(f"  Medyan Bulma Süresi: {median_time:.4f} saniye")
        print(f"  Minimum Bulma Süresi: {min_time:.4f} saniye")
        print(f"  Maksimum Bulma Süresi: {max_time:.4f} saniye")
        print(f"  Varyans: {variance:.6f}")
        print(f"  Standart Sapma: {std_dev:.4f}")
        print("-" * 40)