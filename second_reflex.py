import numpy as np
import matplotlib.pyplot as plt

# 20x20 grid oluştur
def create_grid(size=20, random=True):
    if random:
        return np.random.choice([0, 1], size=(size, size))
    else:  # Glider örneği
        grid = np.zeros((size, size))
        grid[1, 2] = grid[2, 3] = grid[3, 1] = grid[3, 2] = grid[3, 3] = 1
        return grid

# Bir iterasyon çalıştır (Basit Refleks Aracıları - Orijinal Kurallar)
def next_generation(grid):
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            # Komşu sayısını hesapla
            neighbors = 0
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k, l) != (0, 0):  # Merkezi hariç tut
                        ni, nj = (i + k) % size, (j + l) % size  # Dairesel sınırlar
                        neighbors += grid[ni, nj]
            # Orijinal Game of Life kuralları
            if grid[i, j] == 1 and neighbors in [2, 3]:  # Hayatta kalma
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and neighbors == 3:     # Doğum
                new_grid[i, j] = 1
    return new_grid

# Grid’i görselleştir
def plot_grid(grid, title="Game of Life - Basit Refleks Aracıları"):
    plt.imshow(grid, cmap="binary")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Basit Refleks Aracıları ile simülasyon
def run_reflex_agent(grid, iterations=10):
    print("Basit Refleks Aracıları ile simülasyon başlıyor...")
    plot_grid(grid, "Başlangıç Grid’i")
    for i in range(iterations):
        grid = next_generation(grid)
        plot_grid(grid, f"İterasyon {i+1}")

# Çalıştır
if __name__ == "__main__":
    grid = create_grid(random=False)  # Glider ile başla
    run_reflex_agent(grid, iterations=10)