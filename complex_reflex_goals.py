import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import find_peaks

# 50x50 grid oluştur - Daha kompleks bir başlangıç
def create_grid(size=50):
    grid = np.zeros((size, size))
    # Glider
    grid[1, 2] = grid[2, 3] = grid[3, 1] = grid[3, 2] = grid[3, 3] = 1
    # Blinker
    grid[5, 5] = grid[5, 6] = grid[5, 7] = 1
    # Pulsar (döngüsel bir desen)
    grid[10, 12:15] = grid[15, 12:15] = grid[11:14, 10] = grid[11:14, 15] = 1
    # Rastgele bölge
    grid[20:30, 20:30] = np.random.choice([0, 1], size=(10, 10), p=[0.6, 0.4])
    return grid

# Bir iterasyon çalıştır (Basit Refleks Aracıları - Dairesel Sınırlar)
def next_generation(grid):
    size = grid.shape[0]
    new_grid = np.zeros_like(grid)
    for i in range(size):
        for j in range(size):
            neighbors = 0
            for k in [-1, 0, 1]:
                for l in [-1, 0, 1]:
                    if (k, l) != (0, 0):
                        ni, nj = (i + k) % size, (j + l) % size  # Dairesel sınırlar
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

# Grid’in durumunu analiz et
def analyze_grid(live_cells, window=50):
    recent_changes = np.diff(live_cells[-window:])
    mean_change = np.mean(np.abs(recent_changes))
    std_change = np.std(live_cells[-window:])
    
    peaks, _ = find_peaks(live_cells, distance=5)
    if len(peaks) > 2:
        periods = np.diff(peaks)
        if np.std(periods) < 2:
            return "Döngüsel", periods.mean()
    
    if mean_change < 0.1:
        return "Stabil", None
    
    if std_change > 5:
        return "Kaotik", None
    
    return "Belirsiz", None

# Simülasyon ve analiz
def run_reflex_agent(grid, iterations=1000, visualize=True):
    print("Basit Refleks Aracıları ile simülasyon başlıyor...")
    grids = [grid.copy()]
    live_cells = [np.sum(grid)]
    
    start_time = time.time()
    for i in range(iterations):
        grid = next_generation(grid)
        grids.append(grid.copy())
        live_cells.append(np.sum(grid))
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Toplam süre (sadece simülasyon): {duration:.2f} saniye")
    print(f"Başlangıç canlı hücre sayısı: {live_cells[0]}")
    print(f"Son canlı hücre sayısı: {live_cells[-1]}")
    
    state, period = analyze_grid(live_cells)
    print(f"Grid’in durumu: {state}")
    if period:
        print(f"Ortalama döngü uzunluğu: {period:.2f} iterasyon")
    
    if visualize:
        plot_grid(grids[0], "Başlangıç Grid’i (50x50)", save=True)
        for i in [0, 99, 999]:
            plot_grid(grids[i + 1], f"İterasyon {i+1} (50x50)", save=True)
        
        plt.plot(range(iterations + 1), live_cells, label="Canlı Hücre Sayısı")
        plt.xlabel("İterasyon")
        plt.ylabel("Canlı Hücre Sayısı")
        plt.title("Canlı Hücre Sayısı Değişimi")
        plt.legend()
        plt.savefig("live_cells_plot.png")
        plt.show()

# Çalıştır
if __name__ == "__main__":
    grid = create_grid()
    run_reflex_agent(grid, iterations=1000, visualize=True)