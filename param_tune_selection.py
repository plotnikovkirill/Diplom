"""
ПОДБОР ПАРАМЕТРОВ СЕЛЕКЦИИ ЭГА: размер турнира k и размер элиты.

Под замечание научного руководителя: утверждения про k=5 и элиту 10%
должны подкрепляться таблицами. Скрипт даёт две таблицы.

Условия: n = 100, p = 0.15 (граф средней сложности, где разница видна),
по нескольким графам и прогонам для усреднения.

Запуск: python param_tune_selection.py
"""

import time
import statistics
import networkx as nx
import random

try:
    from main_fixed import GeneticAlgorithm
except ImportError:
    from main import GeneticAlgorithm

# ----- параметры -----
N_NODES = 100
P_EDGE = 0.15
GRAPHS = 3            # разных графов для усреднения
RUNS = 4             # прогонов на граф
SEED = 42

TOURNAMENT_SIZES = [2, 3, 5, 7, 10]
ELITE_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25]  # доля от популяции (N=100)

random.seed(SEED)

# Генерируем набор графов один раз, используем для обоих тестов
graphs = []
for g in range(GRAPHS):
    G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED + g)
    G.remove_nodes_from(list(nx.isolates(G)))
    graphs.append(G)

print("=" * 80)
print("ПОДБОР ПАРАМЕТРОВ СЕЛЕКЦИИ ЭГА")
print(f"n = {N_NODES}, p = {P_EDGE}, графов = {GRAPHS}, прогонов на граф = {RUNS}")
print("=" * 80)

# ===================== ТЕСТ 1: РАЗМЕР ТУРНИРА =====================
print("\n" + "=" * 80)
print("ТЕСТ 1: РАЗМЕР ТУРНИРА k (элита фиксирована = 10)")
print("=" * 80)

tour_results = []
for k in TOURNAMENT_SIZES:
    lens, times, gens = [], [], []
    for G in graphs:
        for r in range(RUNS):
            ga = GeneticAlgorithm(G, tournament_k=k, elite_size=10)
            t0 = time.perf_counter()
            path = ga.solve()
            dt = time.perf_counter() - t0
            lens.append(len(path) - 1)
            times.append(dt)
            gens.append(len(ga.history))
    avg = statistics.mean(lens)
    std = statistics.stdev(lens) if len(lens) > 1 else 0.0
    tour_results.append({"k": k, "avg": avg, "std": std,
                         "gens": statistics.mean(gens),
                         "time": statistics.mean(times)})
    print(f"  k={k:>2}: среднее |P|={avg:6.2f}, std={std:5.2f}, "
          f"поколений={statistics.mean(gens):5.0f}, t={statistics.mean(times):.2f}с")

print("\n" + "-" * 80)
print("СВОДНАЯ ТАБЛИЦА (новая Таблица 2.x: размер турнира)")
print("-" * 80)
print(f"{'k':>5} | {'Средняя |P|':>12} | {'Std':>6} | {'Поколений':>10} | {'Время, с':>8}")
print("-" * 80)
for r in tour_results:
    print(f"{r['k']:>5} | {r['avg']:>12.2f} | {r['std']:>6.2f} | "
          f"{r['gens']:>10.0f} | {r['time']:>8.2f}")
best_k = max(tour_results, key=lambda r: r["avg"])
print("-" * 80)
print(f"Лучшее качество: k = {best_k['k']} (среднее |P| = {best_k['avg']:.2f})")

# ===================== ТЕСТ 2: РАЗМЕР ЭЛИТЫ =====================
print("\n" + "=" * 80)
print("ТЕСТ 2: РАЗМЕР ЭЛИТЫ (турнир фиксирован k=5)")
print("=" * 80)

elite_results = []
for frac in ELITE_FRACTIONS:
    elite = max(1, int(N_NODES * frac))   # популяция = 100, поэтому elite = frac*100
    lens, times, gens = [], [], []
    for G in graphs:
        for r in range(RUNS):
            ga = GeneticAlgorithm(G, tournament_k=5, elite_size=elite)
            t0 = time.perf_counter()
            path = ga.solve()
            dt = time.perf_counter() - t0
            lens.append(len(path) - 1)
            times.append(dt)
            gens.append(len(ga.history))
    avg = statistics.mean(lens)
    std = statistics.stdev(lens) if len(lens) > 1 else 0.0
    elite_results.append({"frac": frac, "elite": elite, "avg": avg, "std": std,
                          "gens": statistics.mean(gens),
                          "time": statistics.mean(times)})
    print(f"  элита={elite:>2} ({int(frac*100)}%): среднее |P|={avg:6.2f}, "
          f"std={std:5.2f}, поколений={statistics.mean(gens):5.0f}")

print("\n" + "-" * 80)
print("СВОДНАЯ ТАБЛИЦА (новая Таблица 2.y: размер элиты)")
print("-" * 80)
print(f"{'Элита':>6} | {'% от N':>7} | {'Средняя |P|':>12} | {'Std':>6} | "
      f"{'Поколений':>10} | {'Время, с':>8}")
print("-" * 80)
for r in elite_results:
    print(f"{r['elite']:>6} | {int(r['frac']*100):>6}% | {r['avg']:>12.2f} | "
          f"{r['std']:>6.2f} | {r['gens']:>10.0f} | {r['time']:>8.2f}")
best_e = max(elite_results, key=lambda r: r["avg"])
print("-" * 80)
print(f"Лучшее качество: элита = {best_e['elite']} "
      f"({int(best_e['frac']*100)}%, среднее |P| = {best_e['avg']:.2f})")
print("=" * 80)
