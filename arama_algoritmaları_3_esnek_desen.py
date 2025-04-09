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

# 3x3 bölgenin glider ile farkını hesapla
def check_glider_region(region, glider):
    diff = 0
    for i in range(3):
        for j in range(3):
            if region[i][j] != glider[i][j]:
                diff += 1
    return diff

# Izgarada herhangi bir 3x3 bölgede glider ara (tüm yönler)
def find_glider(grid, glider_variations):
    rows, cols = len(grid), len(grid[0])
    min_diff = float('inf')
    
    # Tüm olası 3x3 bölgeleri tara
    for i in range(rows - 2):
        for j in range(cols - 2):
            region = [row[j:j+3] for row in grid[i:i+3]]
            # Glider’ın 4 yönünü kontrol et
            for glider in glider_variations:
                diff = check_glider_region(region, glider)
                min_diff = min(min_diff, diff)
                if min_diff == 0:  # Tam eşleşme bulundu
                    return 9  # Maksimum puan (3x3’te 9 hücre)
    
    # Tam eşleşme yoksa, en yakın bölgeye göre puan
    return 9 - min_diff

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
def genetic_algorithm(target_variations, rows=5, cols=5, pop_size=200, max_generations=500):
    population = [random_grid(rows, cols) for _ in range(pop_size)]  # Rastgele popülasyon
    start_time = time.time()
    
    for gen in range(max_generations):
        scores = []
        for individual in population:
            evolved = evolve_grid(individual)  # 5 nesil ilerlet
            score = find_glider(evolved, target_variations)  # Glider ara
            scores.append((individual, score))
            if score == 9:  # Tam eşleşme
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
def hill_climbing(target_variations, rows=5, cols=5, max_iterations=100):
    current = random_grid(rows, cols)  # Rastgele başlangıç
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_glider(current_evolved, target_variations)
        
        if current_score == 9:  # Tam eşleşme
            end_time = time.time()
            return current, end_time - start_time
        
        neighbors = get_neighbors(current)  # Komşuları üret
        best_neighbor = current
        best_score = current_score
        
        for neighbor in neighbors:
            neighbor_evolved = evolve_grid(neighbor)
            score = find_glider(neighbor_evolved, target_variations)
            if score > best_score:  # Daha iyi komşu bul
                best_neighbor = neighbor
                best_score = score
        
        if best_score <= current_score:  # İyileşme yoksa dur
            break
        current = best_neighbor
    
    end_time = time.time()
    return current, end_time - start_time

# Benzetimli Tavlama
def simulated_annealing(target_variations, rows=5, cols=5, max_iterations=1000, initial_temp=100):
    current = random_grid(rows, cols)
    temp = initial_temp
    start_time = time.time()
    
    for _ in range(max_iterations):
        current_evolved = evolve_grid(current)
        current_score = find_glider(current_evolved, target_variations)
        
        if current_score == 9:  # Tam eşleşme
            end_time = time.time()
            return current, end_time - start_time
        
        neighbor = random.choice(get_neighbors(current))  # Rastgele bir komşu
        neighbor_evolved = evolve_grid(neighbor)
        neighbor_score = find_glider(neighbor_evolved, target_variations)
        
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
def local_beam_search(target_variations, rows=5, cols=5, beam_width=10, max_iterations=100):
    beams = [random_grid(rows, cols) for _ in range(beam_width)]  # Işınlarla başla
    start_time = time.time()
    
    for _ in range(max_iterations):
        scores = []
        for beam in beams:
            beam_evolved = evolve_grid(beam)
            score = find_glider(beam_evolved, target_variations)
            scores.append((beam, score))
            if score == 9:  # Tam eşleşme
                end_time = time.time()
                return beam, end_time - start_time
        
        candidates = []
        for beam in beams:
            neighbors = get_neighbors(beam)  # Her ışından komşular
            for neighbor in neighbors:
                neighbor_evolved = evolve_grid(neighbor)
                score = find_glider(neighbor_evolved, target_variations)
                candidates.append((neighbor, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)  # En iyileri seç
        beams = [cand[0] for cand in candidates[:beam_width]]
    
    best = max(scores, key=lambda x: x[1])[0]
    end_time = time.time()
    return best, end_time - start_time

# Glider’ın 4 yönü
glider_variations = [
    [[0, 1, 0], [0, 0, 1], [1, 1, 1]],  # Orijinal
    [[1, 0, 0], [1, 1, 0], [1, 0, 1]],  # 90° sağ
    [[1, 1, 1], [1, 0, 0], [0, 1, 0]],  # 180°
    [[0, 1, 0], [1, 1, 0], [0, 0, 1]]   # 90° sol
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
        solution, duration = algo(glider_variations, rows=rows, cols=cols)
        evolved = evolve_grid(solution)
        score = find_glider(evolved, glider_variations)
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