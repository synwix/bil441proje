import numpy as np
import matplotlib.pyplot as plt
import time

# 20x20 grid oluştur
def create_grid(size=20, random=True):
    if random:
        return np.random.choice([0, 1], size=(size, size))
    else:  # Glider örneği
        grid = np.zeros((size, size))
        grid[1, 2] = grid[2, 3] = grid[3, 1] = grid[3, 2] = grid[3, 3] = 1
        return grid

# Bir iterasyon çalıştır (Basit Refleks Aracıları - Sabit Sınırlar)
def next_generation(grid):
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            neighbors = 0
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k, l) != (0, 0):
                        ni, nj = i + k, j + l
                        if 0 <= ni < size and 0 <= nj < size:
                            neighbors += grid[ni, nj]
            if grid[i, j] == 1 and neighbors in [2, 3]:
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and neighbors == 3:
                new_grid[i, j] = 1
    return new_grid

# Grid’i görselleştir ve kaydet
def plot_grid(grid, title="Game of Life - Basit Refleks Aracıları", save=False):
    plt.imshow(grid, cmap="binary")
    plt.title(title)
    plt.axis("off")
    if save:
        plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# Basit Refleks Aracıları ile simülasyon ve analiz
def run_reflex_agent(grid, iterations=100, visualize=True):
    print("Basit Refleks Aracıları ile simülasyon başlıyor...")
    grids = [grid.copy()]  # Tüm grid durumlarını sakla
    live_cells = [np.sum(grid)]  # Canlı hücre sayıları
    
    # Sadece simülasyon süresini ölç
    start_time = time.time()
    for i in range(iterations):
        grid = next_generation(grid)
        grids.append(grid.copy())
        live_cells.append(np.sum(grid))
    end_time = time.time()
    duration = end_time - start_time
    
    # Başarım ölçümleri
    print(f"Toplam süre (sadece simülasyon): {duration:.2f} saniye")
    print(f"Başlangıç canlı hücre sayısı: {live_cells[0]}")
    print(f"Son canlı hücre sayısı: {live_cells[-1]}")
    if live_cells[-1] == live_cells[-2]:
        print("Grid stabil bir duruma ulaştı.")
    
    # Görselleştirme (süreden bağımsız)
    if visualize:
        plot_grid(grids[0], "Başlangıç Grid’i", save=True)
        for i in [0, 9, 99]:  # Örnek iterasyonlar
            plot_grid(grids[i + 1], f"İterasyon {i+1}", save=True)
        
        # Canlı hücre grafiği
        plt.plot(range(iterations + 1), live_cells, label="Canlı Hücre Sayısı")
        plt.xlabel("İterasyon")
        plt.ylabel("Canlı Hücre Sayısı")
        plt.title("Canlı Hücre Sayısı Değişimi")
        plt.legend()
        plt.savefig("live_cells_plot.png")
        plt.show()

# Çalıştır
if __name__ == "__main__":
    grid = create_grid(random=False)  # Glider ile başla
    run_reflex_agent(grid, iterations=100, visualize=True)