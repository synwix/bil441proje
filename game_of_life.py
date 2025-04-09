import numpy as np
import matplotlib.pyplot as plt

# 30x30 grid oluştur
def create_grid(size=30, random=True):
    if random:
        return np.random.choice([0, 1], size=(size, size))
    else:  # Genişletilmiş glider benzeri bir desen
        grid = np.zeros((size, size))
        grid[1, 2] = grid[2, 3] = grid[3, 1] = grid[3, 2] = grid[3, 3] = 1
        grid[2, 1] = 1  # Ek bir hücre ile daha karmaşık bir başlangıç
        return grid

# Bir iterasyon çalıştır (Değiştirilmiş Basit Refleks Aracıları)
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
            # Değiştirilmiş kuralları uygula
            if grid[i, j] == 1 and neighbors in [2, 3, 4]:  # Hayatta kalma
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and neighbors in [3, 4]:  # Doğum
                new_grid[i, j] = 1
    return new_grid

# Grid’i görselleştir
def plot_grid(grid, title="Game of Life - Değiştirilmiş Kurallar"):
    plt.imshow(grid, cmap="binary")
    plt.title(title)
    plt.axis("off")
    plt.show()

# Test çalıştırması
if __name__ == "__main__":
    # Başlangıç grid’i oluştur ve göster
    grid = create_grid(size=30, random=False)  # Genişletilmiş glider ile başla
    plot_grid(grid, "Başlangıç Grid’i (30x30)")
    
    # 5 iterasyon çalıştır ve her birini göster
    for i in range(5):
        grid = next_generation(grid)
        plot_grid(grid, f"İterasyon {i+1} (30x30)")