import random
import time
import copy
import math
import statistics
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# Logging ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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

# Bölgenin belirli bir desenle farkını hesapla
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

# Genetik Algoritmalar (Hiperparametre ile)
def genetic_algorithm(target_pattern, rows=5, cols=5, pop_size=200, max_generations=500, mutation_rate=0.05):
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
            child = mutate(child, rate=mutation_rate)
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

# Yerel Işın Araması (Hiperparametre ile)
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

# Deneme çalıştırma fonksiyonu (paralel için, hiperparametrelerle)
def run_trial(args):
    algo, target_pattern, rows, cols, trial_num, pattern_name, algo_name, key, params = args
    logging.info(f"{pattern_name} - {algo_name} ({key}) [Params: {params}]: Deneme {trial_num + 1}/100 başladı")
    if algo_name == "Genetic Algorithm":
        solution, duration = genetic_algorithm(target_pattern, rows=rows, cols=cols, 
                                              pop_size=params['pop_size'], 
                                              max_generations=params['max_generations'], 
                                              mutation_rate=params['mutation_rate'])
    elif algo_name == "Local Beam Search":
        solution, duration = local_beam_search(target_pattern, rows=rows, cols=cols, 
                                               beam_width=params['beam_width'], 
                                               max_iterations=params['max_iterations'])
    else:
        solution, duration = algo(target_pattern, rows=rows, cols=cols)
    
    evolved = evolve_grid(solution)
    score = find_pattern(evolved, target_pattern)
    max_score = len(target_pattern) * len(target_pattern[0])
    result = (duration, score == max_score)
    if result[1]:
        logging.info(f"{pattern_name} - {algo_name} ({key}) [Params: {params}]: Deneme {trial_num + 1}/100 başarılı")
    else:
        logging.info(f"{pattern_name} - {algo_name} ({key}) [Params: {params}]: Deneme {trial_num + 1}/100 başarısız")
    logging.info(f"{pattern_name} - {algo_name} ({key}) [Params: {params}]: Deneme {trial_num + 1}/100 tamamlandı, süre: {duration:.4f}s")
    return result

# Hedef desenler
boat_pattern = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # Boat
block_pattern = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]  # Block Cluster
glider_pattern = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]  # Glider
lwss_pattern = [[0, 1, 0, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 1]]  # Lightweight Spaceship

# Test edilecek desenler ve isimler
patterns = {
    "Boat": boat_pattern,
    "Block Cluster": block_pattern,
    "Glider": glider_pattern,
    "Lightweight Spaceship": lwss_pattern
}

# Hiperparametre kombinasyonları
ga_params = [
    {"pop_size": 200, "max_generations": 500, "mutation_rate": 0.05},  # Varsayılan
    {"pop_size": 300, "max_generations": 250, "mutation_rate": 0.03},  # Test 1
    {"pop_size": 500, "max_generations": 750, "mutation_rate": 0.1}    # Test 2
]

lbs_params = [
    {"beam_width": 10, "max_iterations": 100},  # Varsayılan
    {"beam_width": 20, "max_iterations": 25},   # Test 1
    {"beam_width": 5, "max_iterations": 75}     # Test 2
]

# Algoritmalar
algorithms = {
    "Genetic Algorithm": genetic_algorithm,
    "Local Beam Search": local_beam_search,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing
}

# 100 deneme testi (paralel ve log ile)
NUM_TRIALS = 100
trial_results = {
    pattern_name: {
        name: {
            f"{rows}x{cols}_{str(params)}": {"times": [], "success": 0, "failure": 0}
            for rows, cols in [(5, 5), (10, 10), (20, 20)]
            for params in (ga_params if name == "Genetic Algorithm" else lbs_params if name == "Local Beam Search" else [{}])
        }
        for name in algorithms
    }
    for pattern_name in patterns
}

with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    for pattern_name, pattern in patterns.items():
        for name, algo in algorithms.items():
            param_list = ga_params if name == "Genetic Algorithm" else lbs_params if name == "Local Beam Search" else [{}]
            for params in param_list:
                for size in [(5, 5), (10, 10), (20, 20)]:
                    rows, cols = size
                    key = f"{rows}x{cols}_{str(params)}"
                    trials = [(algo, pattern, rows, cols, i, pattern_name, name, key, params) for i in range(NUM_TRIALS)]
                    futures = [executor.submit(run_trial, trial) for trial in trials]
                    for future in as_completed(futures):
                        duration, success = future.result()
                        if success:
                            trial_results[pattern_name][name][key]["times"].append(duration)
                            trial_results[pattern_name][name][key]["success"] += 1
                        else:
                            trial_results[pattern_name][name][key]["failure"] += 1

# Sonuçları raporla (ek istatistiklerle)
print("\n100 Deneme İstatistikleri:")
for pattern_name in patterns:
    print(f"\n{pattern_name}:")
    for name in algorithms:
        param_list = ga_params if name == "Genetic Algorithm" else lbs_params if name == "Local Beam Search" else [{}]
        for params in param_list:
            for size in [(5, 5), (10, 10), (20, 20)]:
                rows, cols = size
                key = f"{rows}x{cols}_{str(params)}"
                times = trial_results[pattern_name][name][key]["times"]
                success = trial_results[pattern_name][name][key]["success"]
                failure = trial_results[pattern_name][name][key]["failure"]
                avg_time = sum(times) / len(times) if times else 0
                variance = sum((t - avg_time) ** 2 for t in times) / len(times) if times else 0
                std_dev = math.sqrt(variance) if variance > 0 else 0
                median_time = statistics.median(times) if times else 0
                min_time = min(times) if times else 0
                max_time = max(times) if times else 0
                
                print(f"{name} ({rows}x{cols}) [Params: {params}]:")
                print(f"  Bulma Sayısı: {success}")
                print(f"  Bulamama Sayısı: {failure}")
                print(f"  Ortalama Bulma Süresi (sadece başarılı denemeler): {avg_time:.4f} saniye")
                print(f"  Medyan Bulma Süresi: {median_time:.4f} saniye")
                print(f"  Minimum Bulma Süresi: {min_time:.4f} saniye")
                print(f"  Maksimum Bulma Süresi: {max_time:.4f} saniye")
                print(f"  Varyans: {variance:.6f}")
                print(f"  Standart Sapma: {std_dev:.4f}")
                print("-" * 40)

# Izgarayı yazdırma fonksiyonu (isteğe bağlı)
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()