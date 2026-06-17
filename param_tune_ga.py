"""
ПОДБОР ПАРАМЕТРОВ ЭГА.
Тестирование различных значений размера популяции (N) и числа поколений (G).
Условия: n = 50, p = 0.3.
Запуск: python param_tune_ga.py
"""

import time
import statistics
import networkx as nx
import random

try:
    from main_fixed import GeneticAlgorithm
except ImportError:
    from main import GeneticAlgorithm

# ----- параметры серии -----
# Граф побольше и поразреженнее, чтобы разница между N и G была заметна.
# На p=0.3, n=50 все N>=50 уперлись бы в потолок 49.
N_NODES = 100
P_EDGE = 0.15
RUNS_PER_CONFIG = 4
SEED = 42

# конфигурации (N, G) для перебора
CONFIGS = [
    (30, 500),
    (50, 500),
    (100, 500),
    (200, 500),
    (400, 500),
    (100, 200),
    (100, 1000),
]

random.seed(SEED)

print("=" * 80)
print("ПОДБОР ПАРАМЕТРОВ ЭГА: размер популяции N и число поколений G")
print(f"n = {N_NODES}, p = {P_EDGE}, прогонов на конфигурацию = {RUNS_PER_CONFIG}")
print("=" * 80)

# единый граф для всех конфигураций
G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED)
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер\n")

results = []

for N, GEN in CONFIGS:
    print(f"--- Конфигурация: N={N}, G={GEN} ---")
    lens, times, eff_gens = [], [], []
    for r in range(RUNS_PER_CONFIG):
        # Используем штатный ранний останов (stagnation_limit=80 по умолчанию),
        # чтобы видеть, какая N действительно использует поколения эффективнее.
        ga = GeneticAlgorithm(G, pop_size=N, generations=GEN,
                              elite_size=max(1, N // 10))
        t0 = time.perf_counter()
        path = ga.solve()
        dt = time.perf_counter() - t0
        L = len(path) - 1
        used_gens = len(ga.history)  # сколько реально отработало поколений
        lens.append(L)
        times.append(dt)
        eff_gens.append(used_gens)
        print(f"  Прогон {r + 1}: |P|={L}, поколений={used_gens}, t={dt:.2f}с")
    avg_len = statistics.mean(lens)
    std_len = statistics.stdev(lens) if len(lens) > 1 else 0.0
    avg_t = statistics.mean(times)
    avg_gens = statistics.mean(eff_gens)
    results.append({
        "N": N, "G": GEN, "avg": avg_len, "std": std_len,
        "time": avg_t, "gens": avg_gens
    })
    print(f"  → среднее: {avg_len:.2f}, std: {std_len:.2f}, "
          f"ср. поколений: {avg_gens:.0f}, ср. время: {avg_t:.2f}с\n")

# ----- сводная таблица -----
print("=" * 95)
print("СВОДНАЯ ТАБЛИЦА (Таблица 2.1 в отчёте)")
print("=" * 95)
print(f"{'N':>5} | {'G':>5} | {'Средняя |P|':>12} | {'Std':>5} | "
      f"{'Поколений':>10} | {'Время, с':>8}")
print("-" * 95)
for r in results:
    print(f"{r['N']:>5d} | {r['G']:>5d} | {r['avg']:>12.2f} | "
          f"{r['std']:>5.2f} | {r['gens']:>10.0f} | {r['time']:>8.2f}")
print("=" * 95)

# Поиск разумного компромисса
print("\nАнализ:")
print("  Лучшее качество:", max(results, key=lambda r: r["avg"]))
print("  Минимальное время при качестве ≥ 95% от лучшего:")
best = max(r["avg"] for r in results)
threshold = 0.95 * best
candidates = [r for r in results if r["avg"] >= threshold]
fastest = min(candidates, key=lambda r: r["time"])
print(f"     N={fastest['N']}, G={fastest['G']}, |P|={fastest['avg']:.2f}, "
      f"t={fastest['time']:.2f}с")
