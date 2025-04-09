import numpy as np
import time

def create_grid(size=50):
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[10, 12:15] = grid[15, 12:15] = grid[11:14, 10] = grid[11:14, 15] = 1  # Pulsar
    grid[20, 21] = grid[20, 24] = grid[21, 20] = grid[22, 20] = grid[22, 24] = 1  # LWSS
    grid[23, 20] = grid[23, 23] = grid[21, 24] = grid[22, 23] = 1
    grid[30:45, 30:45] = np.random.choice([0, 1], size=(15, 15), p=[0.7, 0.3])
    attempts = 0
    while has_pentadecathlon(grid) and attempts < 10:
        grid[30:45, 30:45] = np.random.choice([0, 1], size=(15, 15), p=[0.7, 0.3])
        attempts += 1
    return grid

def next_generation(grid):
    size = grid.shape[0]
    neighbors = sum(np.roll(np.roll(grid, i, axis=0), j, axis=1) 
                    for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0))
    birth = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & np.isin(neighbors, [2, 3])
    return (birth | survive).astype(np.uint8)

def has_pentadecathlon(grid):
    pentadecathlon = np.ones((15, 1), dtype=np.uint8)
    size = grid.shape[0]
    for i in range(size - 14):
        for j in range(size):
            if np.array_equal(grid[i:i+15, j:j+1], pentadecathlon):
                return True
    return False

def dfs_search(grid, target_func, max_depth=1000):  # Derinlik 100’e çıkarıldı
    stack = [(grid.copy(), 0)]
    start_time = time.time()
    
    while stack:
        current_grid, depth = stack.pop()
        if depth > max_depth:
            continue
        
        if target_func(current_grid):
            end_time = time.time()
            return True, depth, end_time - start_time, depth + 1
        
        if np.sum(current_grid) == 0:
            end_time = time.time()
            return False, depth, end_time - start_time, depth + 1
        
        next_grid = next_generation(current_grid)
        stack.append((next_grid, depth + 1))
    
    end_time = time.time()
    return False, max_depth, end_time - start_time, max_depth + 1

if __name__ == "__main__":
    grid = create_grid()
    found, depth, duration, visited_count = dfs_search(grid, has_pentadecathlon)
    print("Sonuç:")
    print(f"Pentadecathlon bulundu mu? {found}")
    print(f"Bulunma derinliği: {depth} iterasyon")
    print(f"Arama süresi: {duration:.2f} saniye")
    print(f"Ziyaret edilen durum sayısı: {visited_count}")