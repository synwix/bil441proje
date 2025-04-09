import random
import time
import copy
import math

# Game of Life kurallarını uygulayan fonksiyon (açık sınırlar - toroidal)
def next_generation(grid):
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            # Komşu sayısını hesapla, açık sınırlarla (toroidal)
            neighbors = sum(grid[(i + di) % rows][(j + dj) % cols] 
                            for di in [-1, 0, 1] for dj in [-1, 0, 1] if not (di == 0 and dj == 0))
            # Game of Life kuralları
            if grid[i][j] == 1:
                new_grid[i][j] = 1 if neighbors in [2, 3] else 0
            else:
                new_grid[i][j] = 1 if neighbors == 3 else 0
    return new_grid

# 5 nesil ilerletme fonksiyonu
def evolve_grid(grid, steps=5):
    current = copy.deepcopy(grid)  # Gridin kopyasını al, orijinali değişmesin
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

# Izgarada yalnızca belirli desenleri ara
def find_pattern(grid, patterns):
    rows, cols = len(grid), len(grid[0])
    pattern_rows, pattern_cols = len(patterns[0]), len(patterns[0][0])
    min_diff = float('inf')
    
    # Tüm olası alt bölgeleri tara (desen boyutuna göre)
    for i in range(rows - pattern_rows + 1):
        for j in range(cols - pattern_cols + 1):
            region = [row[j:j+pattern_cols] for row in grid[i:i+pattern_rows]]
            # Yalnızca belirtilen desenleri kontrol et
            for pattern in patterns:
                diff = check_region(region, pattern)
                min_diff = min(min_diff, diff)
                if min_diff == 0:  # Tam eşleşme bulundu
                    return pattern_rows * pattern_cols  # Maksimum puan (desen büyüklüğüne göre)
    
    # Tam eşleşme yoksa, en yakın bölgeye göre puan
    return (pattern_rows * pattern_cols) - min_diff

# Rastgele ızgara oluşturma
def random_grid(rows, cols):
    return [[random.randint(0, 1) for _ in range(cols)] for _ in range(rows)]

# Çaprazlama (crossover) fonksiyonu
def crossover(parent1, parent2):
    split = random.randint(0, len(parent1) - 1)  # Rastgele bir satırdan kes
    child = parent1[:split] + parent2[split:]
    return [row[:] for row in child]  # Yeni bir kopya döndür

# Mutasyon fonksiyonu
def mutate(grid, rate=0.1):
    new_grid = [row[:] for row in grid]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if random.random() < rate:  # Her hücre için mutasyon şansı
                new_grid[i][j] = 1 - new_grid[i][j]
    return new_grid

# Komşu durumlar üretme
def get_neighbors(grid):
    neighbors = []
    rows, cols = len(grid), len(grid[0])
    for i in range(rows):
        for j in range(cols):
            new_grid = [row[:] for row in grid]
            new_grid[i][j] = 1 - new_grid[i][j]  # Tek bir hücreyi tersine çevir
            neighbors.append(new_grid)
    return neighbors

# Genetik Algoritmalar
def genetic_algorithm(target_patterns, rows=5, cols=5, pop_size=200, max_generations=500):
    population = [random_grid(rows, cols) for _ in range(pop_size)]  # Rastgele popülasyon
    start_time = time.time()
    
    for gen in range(max_generations):
        scores = []
        for individual in population:
            evolved = evolve_grid(individual)  # 5 nesil ilerlet
            score = find_pattern(evolved, target_patterns)  # Belirli desenleri ara
            scores.append((individual, score))
            max_score = len(target_patterns[0]) * len(target_patterns[0][0])
            if score == max_score:  # Tam eşleşme
                end_time = time.time()
                return individual, end_time - start_time
        
        # En iyi bireyleri seç (elitizm + rastgele)
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = [scores[i][0] for i in range(int(pop_size * 0.2))] + \
                  random.sample(population, int(pop_size * 0.8))
        
        # Yeni popülasyon oluştur
        new_population = []
        while len(new_population) < pop_size:
            if random.random() < 0.8:  # %80 çaprazlama şansı
                parent1, parent2 = random.sample(selected, 2)
                child = crossover(parent1, parent2)
            else:
                child = random.choice(selected)
            child = mutate(child, rate=0.05)  # Mutasyon uygula
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    best = max(scores, key=lambda x: x[1])[0]  # En iyi bireyi döndür
    end_time = time.time()
    return best, end_time - start_time

# Tepe Tırmanma Araması
def hill_climbing(target_patterns, rows=5, cols=5, max_iterations=100):
    current = random_grid(rows, cols)  # Rastgele başlangıç
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_pattern(current_evolved, target_patterns)
        
        max_score = len(target_patterns[0]) * len(target_patterns[0][0])
        if current_score == max_score:  # Tam eşleşme
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)  # Komşuları üret
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(neighbor)
            score = find_pattern(neighbor_evolved, target_patterns)
            if score > best_score:  # Daha iyi komşu bul
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:  # İyileşme yoksa dur
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

# Benzetimli Tavlama
def simulated_annealing(target_patterns, rows=5, cols=5, max_iterations=1000, initial_temp=100):
    current = random_grid(rows, cols)
    temp = initial_temp
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_pattern(current_evolved, target_patterns)
        
        max_score = len(target_patterns[0]) * len(target_patterns[0][0])
        if current_score == max_score:  # Tam eşleşme
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))  # Rastgele bir komşu
        neighbor_evolved = evolve_grid(neighbor)
        neighbor_score = find_pattern(neighbor_evolved, target_patterns)
        
        if neighbor_score > current_score:  # Daha iyi ise kabul et
            current = neighbor
        else:
            delta = neighbor_score - current_score
            if random.random() < math.exp(delta / temp):  # Olasılıksal kabul
                current = neighbor
        
        temp *= 0.95  # Sıcaklığı azalt
    
    end_time = time.time()
    return current, end_time - start_time

# Yerel Işın Araması
def local_beam_search(target_patterns, rows=5, cols=5, beam_width=10, max_iterations=100):
    beams = [random_grid(rows, cols) for _ in range(beam_width)]  # Işınlarla başla
    start_time = time.time()
    
    for _ in range(max_iterations):
        scores = []
        for beam in beams:
            beam_evolved = evolve_grid(beam)
            score = find_pattern(beam_evolved, target_patterns)
            scores.append((beam, score))
            max_score = len(target_patterns[0]) * len(target_patterns[0][0])
            if score == max_score:  # Tam eşleşme
                end_time = time.time()
                return beam, end_time - start_time
        
        candidates = []
        for beam in beams:
            neighbors = get_neighbors(beam)  # Her ışından komşular
            for neighbor in neighbors:
                neighbor_evolved = evolve_grid(neighbor)
                score = find_pattern(neighbor_evolved, target_patterns)
                candidates.append((neighbor, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)  # En iyileri seç
        beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

# Hedef desen (yalnızca bu spesifik glider aranacak)
target_patterns = [
    [[0, 1, 0], [0, 0, 1], [1, 1, 1]]  # İlk tanımlı glider yönü
    # İleride başka desenler eklemek isterseniz buraya ekleyebilirsiniz:
    # [[1, 0, 0], [1, 1, 0], [1, 0, 1]],  # Örnek: 90° sağ
]

# Izgarayı yazdırma fonksiyonu
def print_grid(grid):
    for row in grid:
        print(" ".join(map(str, row)))
    print()

# Test ve karşılaştırma
algorithms = {
    "Genetic Algorithm": genetic_algorithm,
    "Hill Climbing": hill_climbing,
    "Simulated Annealing": simulated_annealing,
    "Local Beam Search": local_beam_search
}

results = {}
for name, algo in algorithms.items():
    for size in [(5, 5), (10, 10)]:  # 5x5 ve 10x10 için test
        rows, cols = size
        solution, duration = algo(target_patterns, rows=rows, cols=cols)
        evolved = evolve_grid(solution)
        score = find_pattern(evolved, target_patterns)
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