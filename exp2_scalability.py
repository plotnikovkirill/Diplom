"""
ЭКСПЕРИМЕНТ 2. Масштабируемость алгоритмов.
Условия: n ∈ {20, 50, 100, 150, 200}, p = 0.3.
Выводит таблицу зависимости |P| и времени от n.
Запуск: python exp2_scalability.py
"""

import time
import statistics
import networkx as nx
import random


from main_fixed import GeneticAlgorithm, SimulatedAnnealing


# ----- параметры -----
N_VALUES = [20, 50, 100, 150, 200]
P_FIXED = 0.05
GRAPHS_PER_N = 2
RUNS_PER_GRAPH = 3
SEED = 42253

results = {n: {"ga_len": [], "sa_len": [], "ga_t": [], "sa_t": []}
           for n in N_VALUES}

print("=" * 90)
print("СЕРИЯ 2: МАСШТАБИРУЕМОСТЬ")
print(f"p = {P_FIXED}, графов на n = {GRAPHS_PER_N}, прогонов на граф = {RUNS_PER_GRAPH}")
print("=" * 90)

random.seed(SEED)

for n in N_VALUES:
    print(f"\n--- n = {n} ---")
    for g_idx in range(GRAPHS_PER_N):
        G = nx.gnp_random_graph(n, P_FIXED, seed=SEED + n + g_idx)
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"  Граф #{g_idx + 1}: {G.number_of_nodes()} вершин, "
              f"{G.number_of_edges()} рёбер")

        for r in range(RUNS_PER_GRAPH):
            ga = GeneticAlgorithm(G)
            t0 = time.perf_counter()
            ga_path = ga.solve()
            ga_t = time.perf_counter() - t0
            ga_len = len(ga_path) - 1

            sa = SimulatedAnnealing(G)
            t0 = time.perf_counter()
            sa_path = sa.solve()
            sa_t = time.perf_counter() - t0
            sa_len = len(sa_path) - 1

            print(f"    Прогон {r + 1}: ЭГА |P|={ga_len:3d} ({ga_t:6.2f}с) | "
                  f"ИО |P|={sa_len:3d} ({sa_t:6.3f}с)")
            results[n]["ga_len"].append(ga_len)
            results[n]["sa_len"].append(sa_len)
            results[n]["ga_t"].append(ga_t)
            results[n]["sa_t"].append(sa_t)

# ----- итоговая таблица -----
print("\n" + "=" * 95)
print("СВОДНАЯ ТАБЛИЦА (Таблица 2.4 в отчёте)")
print("=" * 95)
print(f"{'n':>5} | {'ЭГА |P|':>8} | {'ИО |P|':>7} | {'ЭГА σ':>6} | "
      f"{'ИО σ':>5} | {'ЭГА t,с':>8} | {'ИО t,с':>8} | {'разрыв t':>10}")
print("-" * 95)

for n in N_VALUES:
    r = results[n]
    ga_avg = statistics.mean(r["ga_len"])
    sa_avg = statistics.mean(r["sa_len"])
    ga_std = statistics.stdev(r["ga_len"]) if len(r["ga_len"]) > 1 else 0.0
    sa_std = statistics.stdev(r["sa_len"]) if len(r["sa_len"]) > 1 else 0.0
    ga_t = statistics.mean(r["ga_t"])
    sa_t = statistics.mean(r["sa_t"])
    ratio = ga_t / sa_t if sa_t > 0 else float("inf")

    print(f"{n:>5d} | {ga_avg:>8.1f} | {sa_avg:>7.1f} | {ga_std:>6.2f} | "
          f"{sa_std:>5.2f} | {ga_t:>8.3f} | {sa_t:>8.4f} | x{ratio:>7.1f}")

print("=" * 95)
print("Готово.")
