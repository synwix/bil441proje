import random
import time
import copy
import math
import statistics
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from functools import lru_cache

# Game of Life kurallarını uygulayan fonksiyon (NumPy ile vektörize)
def next_generation(grid):
    grid = np.array(grid, dtype=np.int8)
    rows, cols = grid.shape
    new_grid = np.zeros((rows, cols), dtype=np.int8)
    
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            rolled = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
            new_grid += rolled
    
    new_grid = np.where((grid == 1) & ((new_grid == 2) | (new_grid == 3)), 1,
                        np.where((grid == 0) & (new_grid == 3), 1, 0))
    return new_grid.tolist()

# Belirtilen nesil sayısı kadar ilerletme fonksiyonu (ön bellekleme ile)
@lru_cache(maxsize=1000)
def evolve_grid(grid_tuple, steps=5):
    grid = list(map(list, grid_tuple))
    current = copy.deepcopy(grid)
    for _ in range(steps):
        current = next_generation(current)
    return current

# Bölgenin desenle farkını hesapla
def check_region(region, pattern):
    diff = 0
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if region[i][j] != pattern[i][j]:
                diff += 1
    return diff

# Izgarada belirli desenleri ara (tam eşleşme zorunlu)
def find_pattern(grid, patterns):
    rows, cols = len(grid), len(grid[0])
    pattern_rows, pattern_cols = len(patterns[0]), len(patterns[0][0])
    max_score = pattern_rows * pattern_cols
    
    if pattern_rows > rows or pattern_cols > cols:
        raise ValueError("Hedef desen, ızgara boyutlarını aşıyor!")
    
    for i in range(rows - pattern_rows + 1):
        for j in range(cols - pattern_cols + 1):
            region = [row[j:j+pattern_cols] for row in grid[i:i+pattern_rows]]
            for pattern in patterns:
                if all(region[r][c] == pattern[r][c] for r in range(pattern_rows) 
                       for c in range(pattern_cols)):
                    return max_score
    
    min_diff = float('inf')
    for i in range(rows - pattern_rows + 1):
        for j in range(cols - pattern_cols + 1):
            region = [row[j:j+pattern_cols] for row in grid[i:i+pattern_rows]]
            for pattern in patterns:
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

# Birey değerlendirme fonksiyonu (paralel için)
def evaluate_individual(args):
    individual, target_patterns = args
    evolved = evolve_grid(tuple(map(tuple, individual)))
    score = find_pattern(evolved, target_patterns)
    return individual, score

# Genetik Algoritmalar (Paralel İç Döngü)
def genetic_algorithm(target_patterns, rows, cols, pop_size=200, max_generations=500):
    pattern_rows, pattern_cols = len(target_patterns[0]), len(target_patterns[0][0])
    if pattern_rows > rows or pattern_cols > cols:
        raise ValueError("Hedef desen, ızgara boyutlarını aşıyor!")
    
    population = [random_grid(rows, cols) for _ in range(pop_size)]
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for gen in range(max_generations):
            futures = [executor.submit(evaluate_individual, (ind, target_patterns)) for ind in population]
            scores = [future.result() for future in as_completed(futures)]
            
            max_score = len(target_patterns[0]) * len(target_patterns[0][0])
            for ind, score in scores:
                if score == max_score:
                    return ind, time.time() - start_time
            
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
def hill_climbing(target_patterns, rows, cols, max_iterations=100):
    pattern_rows, pattern_cols = len(target_patterns[0]), len(target_patterns[0][0])
    if pattern_rows > rows or pattern_cols > cols:
        raise ValueError("Hedef desen, ızgara boyutlarını aşıyor!")
    
    current = random_grid(rows, cols)
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(tuple(map(tuple, current)))
        current_score = find_pattern(current_evolved, target_patterns)
        
        max_score = len(target_patterns[0]) * len(target_patterns[0][0])
        if current_score == max_score:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(tuple(map(tuple, neighbor)))
            score = find_pattern(neighbor_evolved, target_patterns)
            if score > best_score:
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

# Benzetimli Tavlama
def simulated_annealing(target_patterns, rows, cols, max_iterations=1000, initial_temp=100):
    pattern_rows, pattern_cols = len(target_patterns[0]), len(target_patterns[0][0])
    if pattern_rows > rows or pattern_cols > cols:
        raise ValueError("Hedef desen, ızgara boyutlarını aşıyor!")
    
    current = random_grid(rows, cols)
    temp = initial_temp
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(tuple(map(tuple, current)))
        current_score = find_pattern(current_evolved, target_patterns)
        
        max_score = len(target_patterns[0]) * len(target_patterns[0][0])
        if current_score == max_score:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))
        neighbor_evolved = evolve_grid(tuple(map(tuple, neighbor)))
        neighbor_score = find_pattern(neighbor_evolved, target_patterns)
        
        if neighbor_score > current_score:
            current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < math.exp(delta / temp):
                current = neighbor
        
        temp *= 0.95
    
    end_time = time.time()
    return current, end_time - start_time

# Yerel Işın Araması (Paralel İç Döngü)
def local_beam_search(target_patterns, rows, cols, beam_width=10, max_iterations=100):
    pattern_rows, pattern_cols = len(target_patterns[0]), len(target_patterns[0][0])
    if pattern_rows > rows or pattern_cols > cols:
        raise ValueError("Hedef desen, ızgara boyutlarını aşıyor!")
    
    beams = [random_grid(rows, cols) for _ in range(beam_width)]
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for _ in range(max_iterations):
            futures = [executor.submit(evaluate_individual, (beam, target_patterns)) for beam in beams]
            scores = [future.result() for future in as_completed(futures)]
            
            max_score = len(target_patterns[0]) * len(target_patterns[0][0])
            for beam, score in scores:
                if score == max_score:
                    return beam, time.time() - start_time
            
            candidates = []
            for beam in beams:
                neighbors = get_neighbors(beam)
                neighbor_futures = [executor.submit(evaluate_individual, (n, target_patterns)) for n in neighbors]
                candidates.extend([future.result() for future in as_completed(neighbor_futures)])
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

# Hedef desen (Boat)
target_patterns = [
    [[0, 1, 0], 
     [1, 0, 1], 
     [0, 1, 0]]
]

# Izgarayı yazdırma fonksiyonu
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

# Tek deneme testi fonksiyonu
def run_single_test(algo, target_patterns, rows, cols):
    try:
        solution, duration = algo(target_patterns, rows=rows, cols=cols)
        evolved = evolve_grid(tuple(map(tuple, solution)))
        score = find_pattern(evolved, target_patterns)
        print(f"{algo.__name__} ({rows}x{cols}):")
        print("Başlangıç Izgara:")
        print_grid(solution)
        print("5. Nesil:")
        print_grid(evolved)
        print(f"Süre: {duration:.4f} saniye")
        print(f"Puan: {score}")
        print("-" * 20)
        return solution, duration, score
    except ValueError as e:
        print(f"{algo.__name__} ({rows}x{cols}): Hata: {e}")
        return None, None, None

# 100 deneme testi fonksiyonu (paralel)
def run_trial(args):
    algo, target_patterns, rows, cols = args
    try:
        solution, duration = algo(target_patterns, rows=rows, cols=cols)
        evolved = evolve_grid(tuple(map(tuple, solution)))
        score = find_pattern(evolved, target_patterns)
        max_score = len(target_patterns[0]) * len(target_patterns[0][0])
        return duration, score == max_score
    except ValueError:
        return None, False

# Ana test fonksiyonu
def main():
    algorithms = {
        "Genetic Algorithm": genetic_algorithm,
        "Hill Climbing": hill_climbing,
        "Simulated Annealing": simulated_annealing,
        "Local Beam Search": local_beam_search
    }

    # Test edilecek ızgara boyutları
    grid_sizes = [(5, 5), (10, 10)]

    # Tek deneme testi
    results = {}
    for name, algo in algorithms.items():
        for rows, cols in grid_sizes:
            solution, duration, score = run_single_test(algo, target_patterns, rows, cols)
            results[f"{name} ({rows}x{cols})"] = (solution, duration, score)

    print("Süre Karşılaştırması:")
    for name, (solution, duration, score) in results.items():
        if solution is not None:
            print(f"{name}: {duration:.4f} saniye (Puan: {score})")
        else:
            print(f"{name}: Çalışmadı")

    # 100 deneme testi
    NUM_TRIALS = 100
    trial_results = {
        name: {
            f"{rows}x{cols}": {"times": [], "success": 0, "failure": 0}
            for rows, cols in grid_sizes
        }
        for name in algorithms
    }

    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        for name, algo in algorithms.items():
            for rows, cols in grid_sizes:
                key = f"{rows}x{cols}"
                trials = [(algo, target_patterns, rows, cols) for _ in range(NUM_TRIALS)]
                futures = [executor.submit(run_trial, trial) for trial in trials]
                results = [future.result() for future in as_completed(futures)]
                
                for duration, success in results:
                    if duration is not None:
                        if success:
                            trial_results[name][key]["times"].append(duration)
                            trial_results[name][key]["success"] += 1
                        else:
                            trial_results[name][key]["failure"] += 1
                    else:
                        trial_results[name][key]["failure"] += 1

    # Sonuçları raporla
    print("\n100 Deneme İstatistikleri:")
    for name in algorithms:
        for rows, cols in grid_sizes:
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

if __name__ == '__main__':
    main()