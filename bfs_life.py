import numpy as np
import time

# 50x50 grid oluştur - Glider olmadan
def create_grid(size=50):
    grid = np.zeros((size, size))
    # Blinker
    grid[5, 5] = grid[5, 6] = grid[5, 7] = 1
    # Pulsar
    grid[10, 12:15] = grid[15, 12:15] = grid[11:14, 10] = grid[11:14, 15] = 1
    # Lightweight Spaceship (LWSS)
    grid[20, 21] = grid[20, 24] = grid[21, 20] = grid[22, 20] = grid[22, 24] = 1
    grid[23, 20] = grid[23, 23] = grid[21, 24] = grid[22, 23] = 1
    # Büyük rastgele bölge
    grid[30:45, 30:45] = np.random.choice([0, 1], size=(15, 15), p=[0.5, 0.5])
    return grid

# Bir iterasyon çalıştır (Reflex’ten)
def next_generation(grid):
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            neighbors = 0
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k, l) != (0, 0):
                        ni, nj = (i + k) % size, (j + l) % size
                        neighbors += grid[ni, nj]
            if grid[i, j] == 1 and neighbors in [2, 3, 4]:
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and neighbors in [3, 4]:
                new_grid[i, j] = 1
    return new_grid


import numpy as np
import time
from collections import deque

# create_grid ve next_generation aynı kalacak (yukarıdaki gibi)

# Stabil durumu kontrol et
def is_stable(grid, prev_grid):
    return np.array_equal(grid, prev_grid)

# BFS ile stabilite arama
def bfs_search(grid, max_steps=1000):
    visited = set()
    queue = deque([(grid.copy(), 0, None)])  # (grid, step, prev_grid)
    start_time = time.time()
    
    while queue:
        current_grid, step, prev_grid = queue.popleft()
        grid_key = tuple(current_grid.flatten())
        if grid_key in visited or step > max_steps:
            continue
        
        visited.add(grid_key)
        if prev_grid is not None and is_stable(current_grid, prev_grid):
            end_time = time.time()
            return step, end_time - start_time, len(visited)
        
        next_grid = next_generation(current_grid)
        queue.append((next_grid, step + 1, current_grid))
    
    end_time = time.time()
    return max_steps, end_time - start_time, len(visited)

# Çalıştır
if __name__ == "__main__":
    grid = create_grid()
    steps, duration, visited_count = bfs_search(grid)
    print(f"Stabil duruma ulaşma adımı: {steps} iterasyon")
    print(f"Arama süresi: {duration:.2f} saniye")
    print(f"Ziyaret edilen durum sayısı: {visited_count}")