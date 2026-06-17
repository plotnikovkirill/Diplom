"""
ЭКСПЕРИМЕНТ 1. Влияние плотности графа.
Условия: n = 50, p ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}.
На каждое значение p — несколько графов и несколько запусков на каждом.
Выводит таблицу: p | ЭГА |P| | ИО |P| | ЭГА σ | ИО σ | ЭГА t | ИО t.
Запуск: python exp1_density.py
"""

import time
import statistics
import networkx as nx

from main_fixed import GeneticAlgorithm, SimulatedAnnealing

# ----- параметры серии -----
N_NODES = 50
P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GRAPHS_PER_P = 3       # сколько разных графов на каждое p
RUNS_PER_GRAPH = 4     # сколько прогонов алгоритма на каждом графе
# Итого 12 точек на условие. Можно поднять, если есть время.
SEED = 4256

# ----- хранилище -----
results = {p: {"ga_len": [], "sa_len": [], "ga_t": [], "sa_t": []}
           for p in P_VALUES}


def run_algo(solver_cls, graph):
    """Один прогон алгоритма. Возвращает (длина, время)."""
    solver = solver_cls(graph)
    t0 = time.perf_counter()
    path = solver.solve()
    dt = time.perf_counter() - t0
    return len(path) - 1, dt


print("=" * 90)
print("СЕРИЯ 1: ВЛИЯНИЕ ПЛОТНОСТИ ГРАФА")
print(f"n = {N_NODES}, графов на p = {GRAPHS_PER_P}, прогонов на граф = {RUNS_PER_GRAPH}")
print("=" * 90)

import random
random.seed(SEED)

for p in P_VALUES:
    print(f"\n--- p = {p} ---")
    for g_idx in range(GRAPHS_PER_P):
        G = nx.gnp_random_graph(N_NODES, p, seed=SEED + g_idx)
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"  Граф #{g_idx + 1}: {G.number_of_nodes()} вершин, "
              f"{G.number_of_edges()} рёбер")

        for r in range(RUNS_PER_GRAPH):
            ga_len, ga_t = run_algo(GeneticAlgorithm, G)
            sa_len, sa_t = run_algo(SimulatedAnnealing, G)
            print(f"    Прогон {r + 1}: ЭГА |P|={ga_len:3d} (t={ga_t:.2f}с) | "
                  f"ИО |P|={sa_len:3d} (t={sa_t:.3f}с)")
            results[p]["ga_len"].append(ga_len)
            results[p]["sa_len"].append(sa_len)
            results[p]["ga_t"].append(ga_t)
            results[p]["sa_t"].append(sa_t)

# ----- итоговая таблица -----
print("\n" + "=" * 90)
print("СВОДНАЯ ТАБЛИЦА (для занесения в отчёт)")
print("=" * 90)
print(f"{'p':>5} | {'ЭГА |P|':>8} | {'ИО |P|':>7} | {'ЭГА σ':>6} | "
      f"{'ИО σ':>5} | {'ЭГА t,с':>8} | {'ИО t,с':>8} | {'Победитель':>10}")
print("-" * 90)

for p in P_VALUES:
    r = results[p]
    ga_avg = statistics.mean(r["ga_len"])
    sa_avg = statistics.mean(r["sa_len"])
    ga_std = statistics.stdev(r["ga_len"]) if len(r["ga_len"]) > 1 else 0.0
    sa_std = statistics.stdev(r["sa_len"]) if len(r["sa_len"]) > 1 else 0.0
    ga_t = statistics.mean(r["ga_t"])
    sa_t = statistics.mean(r["sa_t"])

    if abs(ga_avg - sa_avg) < 0.5:
        winner = "ничья"
    elif ga_avg > sa_avg:
        winner = "ЭГА"
    else:
        winner = "ИО"

    print(f"{p:>5.1f} | {ga_avg:>8.1f} | {sa_avg:>7.1f} | {ga_std:>6.2f} | "
          f"{sa_std:>5.2f} | {ga_t:>8.3f} | {sa_t:>8.4f} | {winner:>10}")

print("=" * 90)
print("Готово. Перенесите числа в Таблицу 2.3 отчёта.")
