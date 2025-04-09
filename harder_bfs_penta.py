import numpy as np
import time
from collections import deque

# Küçük grid ve basit başlangıç
def create_grid(size=20):
    grid = np.zeros((size, size), dtype=np.uint8)
    # Küçük pulsar
    grid[5, 6:9] = grid[8, 6:9] = grid[6:8, 5] = grid[6:8, 9] = 1
    # Küçük rastgele bölge
    grid[10:15, 10:15] = np.random.choice([0, 1], size=(5, 5), p=[0.8, 0.2])
    return grid

def next_generation(grid):
    size = grid.shape[0]
    neighbors = sum(np.roll(np.roll(grid, i, axis=0), j, axis=1) 
                    for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0))
    birth = (grid == 0) & (neighbors == 3)
    survive = (grid == 1) & np.isin(neighbors, [2, 3])
    return (birth | survive).astype(np.uint8)

def is_stable(grid, prev_grid):
    return np.array_equal(grid, prev_grid)

def bfs_search(grid, max_steps=100):
    queue = deque([(grid.copy(), 0, None)])
    visited = set()
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
        
        if np.sum(current_grid) == 0:
            end_time = time.time()
            return step, end_time - start_time, len(visited)
        
        next_grid = next_generation(current_grid)
        queue.append((next_grid, step + 1, current_grid))
    
    end_time = time.time()
    return max_steps, end_time - start_time, len(visited)

if __name__ == "__main__":
    grid = create_grid()
    steps, duration, visited_count = bfs_search(grid)
    print("Sonuç:")
    print(f"Stabil duruma ulaşma adımı: {steps} iterasyon")
    print(f"Arama süresi: {duration:.2f} saniye")
    print(f"Ziyaret edilen durum sayısı: {visited_count}")