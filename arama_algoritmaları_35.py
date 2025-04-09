import random
import time
import copy
import math
import statistics
import heapq

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

# Açgözlü En İyi Öncelikli Arama (Greedy Best-First Search)
def greedy_best_first_search(target_pattern, rows, cols):
    def heuristic(grid):
        evolved = evolve_grid(grid)
        return find_pattern(evolved, target_pattern)

    start_time = time.time()
    start_grid = random_grid(rows, cols)
    visited = set()
    pq = [(heuristic(start_grid), start_grid)]
    heapq.heapify(pq)
    
    while pq:
        _, current = heapq.heappop(pq)
        current_tuple = tuple(map(tuple, current))
        if current_tuple in visited:
            continue
        visited.add(current_tuple)
        
        evolved = evolve_grid(current)
        score = find_pattern(evolved, target_pattern)
        max_score = len(target_pattern) * len(target_pattern[0])
        if score == max_score:
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)
        for neighbor in neighbors:
            neighbor_tuple = tuple(map(tuple, neighbor))
            if neighbor_tuple not in visited:
                heapq.heappush(pq, (heuristic(neighbor), neighbor))
    
    end_time = time.time()
    return current, end_time - start_time

# Hedef desenler
boat_pattern = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]  # Boat
block_pattern = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]  # Block Cluster
glider_pattern = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]  # Glider

# Test edilecek desenler ve isimler
patterns = {
    "Boat": boat_pattern,
    "Block Cluster": block_pattern,
    "Glider": glider_pattern
}

# Izgarayı yazdırma fonksiyonu
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

# Tek deneme testi
results = {}
for pattern_name, pattern in patterns.items():
    for size in [(5, 5), (10, 10), (20, 20)]:
        rows, cols = size
        solution, duration = greedy_best_first_search(pattern, rows, cols)
        evolved = evolve_grid(solution)
        score = find_pattern(evolved, pattern)
        results[f"{pattern_name} - Greedy Best-First Search ({rows}x{cols})"] = (solution, duration, score)
        
        print(f"{pattern_name} - Greedy Best-First Search ({rows}x{cols}):")
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

# 100 deneme testi
NUM_TRIALS = 100
trial_results = {
    pattern_name: {
        f"{rows}x{cols}": {"times": [], "success": 0, "failure": 0}
        for rows, cols in [(5, 5), (10, 10), (20, 20)]
    }
    for pattern_name in patterns
}

for pattern_name, pattern in patterns.items():
    for size in [(5, 5), (10, 10), (20, 20)]:
        rows, cols = size
        key = f"{rows}x{cols}"
        for _ in range(NUM_TRIALS):
            solution, duration = greedy_best_first_search(pattern, rows, cols)
            evolved = evolve_grid(solution)
            score = find_pattern(evolved, pattern)
            max_score = len(pattern) * len(pattern[0])
            if score == max_score:
                trial_results[pattern_name][key]["times"].append(duration)
                trial_results[pattern_name][key]["success"] += 1
            else:
                trial_results[pattern_name][key]["failure"] += 1

# Sonuçları raporla
print("\n100 Deneme İstatistikleri:")
for pattern_name in patterns:
    print(f"\n{pattern_name}:")
    for size in [(5, 5), (10, 10), (20, 20)]:
        rows, cols = size
        key = f"{rows}x{cols}"
        times = trial_results[pattern_name][key]["times"]
        success = trial_results[pattern_name][key]["success"]
        failure = trial_results[pattern_name][key]["failure"]
        avg_time = sum(times) / len(times) if times else 0
        variance = sum((t - avg_time) ** 2 for t in times) / len(times) if times else 0
        std_dev = math.sqrt(variance) if variance > 0 else 0
        median_time = statistics.median(times) if times else 0
        min_time = min(times) if times else 0
        max_time = max(times) if times else 0
        
        print(f"Greedy Best-First Search ({key}):")
        print(f"  Bulma Sayısı: {success}")
        print(f"  Bulamama Sayısı: {failure}")
        print(f"  Ortalama Bulma Süresi (sadece başarılı denemeler): {avg_time:.4f} saniye")
        print(f"  Medyan Bulma Süresi: {median_time:.4f} saniye")
        print(f"  Minimum Bulma Süresi: {min_time:.4f} saniye")
        print(f"  Maksimum Bulma Süresi: {max_time:.4f} saniye")
        print(f"  Varyans: {variance:.6f}")
        print(f"  Standart Sapma: {std_dev:.4f}")
        print("-" * 40)