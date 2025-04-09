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

# Benzetimli Tavlama (Hiperparametre ile)
def simulated_annealing(target_pattern, rows=5, cols=5, max_iterations=1000, initial_temp=100, cooling_rate=0.95):
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
        
        temp *= cooling_rate
    
    end_time = time.time()
    return current, end_time - start_time

# Deneme çalıştırma fonksiyonu (paralel için, hiperparametrelerle)
def run_trial(args):
    target_pattern, rows, cols, trial_num, pattern_name, algo_name, key, params = args
    logging.info(f"{pattern_name} - {algo_name} ({key}) [Params: {params}]: Deneme {trial_num + 1}/100 başladı")
    solution, duration = simulated_annealing(target_pattern, rows=rows, cols=cols, 
                                             max_iterations=params['max_iterations'], 
                                             initial_temp=params['initial_temp'], 
                                             cooling_rate=params['cooling_rate'])
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
sa_params = [
    {"max_iterations": 1000, "initial_temp": 100, "cooling_rate": 0.95},  # Varsayılan
    {"max_iterations": 1500, "initial_temp": 100, "cooling_rate": 0.95},  # 1: Sadece max_iterations
    {"max_iterations": 1000, "initial_temp": 200, "cooling_rate": 0.95},  # 2: Sadece initial_temp
    {"max_iterations": 1000, "initial_temp": 100, "cooling_rate": 0.90},  # 3: Sadece cooling_rate
    {"max_iterations": 1500, "initial_temp": 200, "cooling_rate": 0.95},  # 4: max_iterations + initial_temp
    {"max_iterations": 1500, "initial_temp": 100, "cooling_rate": 0.90},  # 5: max_iterations + cooling_rate
    {"max_iterations": 1000, "initial_temp": 200, "cooling_rate": 0.90},  # 6: initial_temp + cooling_rate
    {"max_iterations": 1500, "initial_temp": 200, "cooling_rate": 0.90},  # 7: Hepsi
    {"max_iterations": 500, "initial_temp": 100, "cooling_rate": 0.95},   # 8: max_iterations azalt
    {"max_iterations": 2000, "initial_temp": 300, "cooling_rate": 0.85}   # 9: Hepsi maksimum
]

# Algoritma
algorithms = {"Simulated Annealing": simulated_annealing}

# 100 deneme testi (paralel ve log ile)
NUM_TRIALS = 100
trial_results = {
    pattern_name: {
        name: {
            f"{rows}x{cols}_{str(params)}": {"times": [], "success": 0, "failure": 0}
            for rows, cols in [(5, 5), (10, 10), (20, 20)]
            for params in sa_params
        }
        for name in algorithms
    }
    for pattern_name in patterns
}

with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
    for pattern_name, pattern in patterns.items():
        for name, algo in algorithms.items():
            for params in sa_params:
                for size in [(5, 5), (10, 10), (20, 20)]:
                    rows, cols = size
                    key = f"{rows}x{cols}_{str(params)}"
                    trials = [(pattern, rows, cols, i, pattern_name, name, key, params) for i in range(NUM_TRIALS)]
                    futures = [executor.submit(run_trial, trial) for trial in trials]
                    for future in as_completed(futures):
                        duration, success = future.result()
                        if success:
                            trial_results[pattern_name][name][key]["times"].append(duration)
                            trial_results[pattern_name][name][key]["success"] += 1
                        else:
                            trial_results[pattern_name][name][key]["failure"] += 1

# Sonuçları hem ekrana yazdır hem dosyaya kaydet
print("\n100 Deneme İstatistikleri:")
with open("sa_hyperparam_results.txt", "w") as f:
    f.write("100 Deneme İstatistikleri:\n")
    for pattern_name in patterns:
        print(f"\n{pattern_name}:")
        f.write(f"\n{pattern_name}:\n")
        for name in algorithms:
            for params in sa_params:
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
                    
                    result_str = (
                        f"{name} ({rows}x{cols}) [Params: {params}]:\n"
                        f"  Bulma Sayısı: {success}\n"
                        f"  Bulamama Sayısı: {failure}\n"
                        f"  Ortalama Bulma Süresi (sadece başarılı denemeler): {avg_time:.4f} saniye\n"
                        f"  Medyan Bulma Süresi: {median_time:.4f} saniye\n"
                        f"  Minimum Bulma Süresi: {min_time:.4f} saniye\n"
                        f"  Maksimum Bulma Süresi: {max_time:.4f} saniye\n"
                        f"  Varyans: {variance:.6f}\n"
                        f"  Standart Sapma: {std_dev:.4f}\n"
                        f"{'-' * 40}\n"
                    )
                    print(result_str)
                    f.write(result_str)