"""
ПОДБОР ПАРАМЕТРОВ МЕТОДА ИМИТАЦИИ ОТЖИГА.
Тестирование различных значений коэффициента охлаждения α.
Условия: n = 50, p = 0.3.
Запуск: python param_tune_sa.py
"""

import time
import statistics
import networkx as nx
import random

try:
    from main_fixed import SimulatedAnnealing
except ImportError:
    from main import SimulatedAnnealing

# ----- параметры -----
# Берём граф потяжелее (n=100, p=0.15), чтобы разница между α была видна.
# На p=0.3, n=50 после починки ИО все α показывают потолок 49 — тест не различает.
N_NODES = 100
P_EDGE = 0.15
RUNS_PER_ALPHA = 10
SEED = 42

ALPHAS = [0.70, 0.80, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99]

random.seed(SEED)

print("=" * 80)
print("ПОДБОР ПАРАМЕТРОВ ИО: коэффициент охлаждения α")
print(f"n = {N_NODES}, p = {P_EDGE}, прогонов на α = {RUNS_PER_ALPHA}")
print("T₀ = 2000, итераций на уровень = 100")
print("=" * 80)

G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED)
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер\n")

results = []

for alpha in ALPHAS:
    print(f"--- α = {alpha} ---")
    lens, times = [], []
    for r in range(RUNS_PER_ALPHA):
        sa = SimulatedAnnealing(G, initial_temp=2000, cooling_rate=alpha,
                                iterations_per_temp=100)
        t0 = time.perf_counter()
        path = sa.solve()
        dt = time.perf_counter() - t0
        L = len(path) - 1
        lens.append(L)
        times.append(dt)
        if r < 5 or r == RUNS_PER_ALPHA - 1:
            print(f"  Прогон {r + 1:2d}: |P|={L}, t={dt:.4f}с")
    avg_len = statistics.mean(lens)
    std_len = statistics.stdev(lens) if len(lens) > 1 else 0.0
    avg_t = statistics.mean(times)
    results.append({"alpha": alpha, "avg": avg_len, "std": std_len, "time": avg_t})
    print(f"  → среднее: {avg_len:.2f}, std: {std_len:.2f}, ср. время: {avg_t:.4f}с\n")

# ----- сводная таблица -----
print("=" * 70)
print("СВОДНАЯ ТАБЛИЦА (Таблица 2.2 в отчёте)")
print("=" * 70)
print(f"{'α':>6} | {'Средняя |P|':>12} | {'Std':>5} | {'Время, с':>10}")
print("-" * 70)
for r in results:
    print(f"{r['alpha']:>6.2f} | {r['avg']:>12.2f} | {r['std']:>5.2f} | "
          f"{r['time']:>10.4f}")
print("=" * 70)

# подсветка оптимума
best = max(results, key=lambda r: r["avg"])
print(f"\nЛучшее качество: α = {best['alpha']}, среднее |P| = {best['avg']:.2f}")

# компромиссное значение
threshold = 0.99 * best["avg"]
candidates = [r for r in results if r["avg"] >= threshold]
fastest = min(candidates, key=lambda r: r["time"])
print(f"Минимальное время при качестве ≥ 99% от лучшего:")
print(f"  α = {fastest['alpha']}, |P| = {fastest['avg']:.2f}, "
      f"t = {fastest['time']:.4f}с")
