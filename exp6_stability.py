"""
ЭКСПЕРИМЕНТ 6 (ПЕРЕРАБОТАННЫЙ). Устойчивость алгоритмов.

Замечание научного руководителя: тест на ОДНОМ графе может отражать
особенность именно его конфигурации. Поэтому теперь устойчивость
проверяется на НЕСКОЛЬКИХ независимых графах одной размерности.

Условия: n = 100, p = 0.2, GRAPHS разных графов, RUNS запусков на каждом.
Для каждого графа считаются среднее, std, min/max, 95% ДИ, U-критерий.
В конце выводится сводка по всем графам: стабилен ли разрыв ЭГА/ИО.

Запуск: python exp6_stability.py
"""

import time
import math
import statistics
import networkx as nx
import random

try:
    from main_fixed import GeneticAlgorithm, SimulatedAnnealing
except ImportError:
    from main import GeneticAlgorithm, SimulatedAnnealing

# ----- параметры -----
N_NODES = 100
P_EDGE = 0.2
GRAPHS = 5          # сколько разных графов протестировать
RUNS = 20           # запусков каждого алгоритма на каждом графе
SEED = 42

random.seed(SEED)


# ----- 95% доверительный интервал по t-распределению -----
T_CRIT = {
    5: 2.571, 10: 2.228, 15: 2.131, 19: 2.093, 20: 2.086, 25: 2.060,
    29: 2.045, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000, 100: 1.984,
}

def t_critical(df):
    for k in sorted(T_CRIT.keys()):
        if df <= k:
            return T_CRIT[k]
    return 1.96

def confidence_interval_95(data):
    n = len(data)
    if n < 2:
        return (data[0], data[0])
    mean = statistics.mean(data)
    se = statistics.stdev(data) / math.sqrt(n)
    t = t_critical(n - 1)
    return (mean - t * se, mean + t * se)


def mann_whitney(x, y):
    """U-критерий Манна-Уитни с нормальной аппроксимацией."""
    nx_, ny = len(x), len(y)
    combined = [(v, "x") for v in x] + [(v, "y") for v in y]
    combined.sort(key=lambda t: t[0])
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rx = sum(ranks[k] for k, (_, lbl) in enumerate(combined) if lbl == "x")
    U = rx - nx_ * (nx_ + 1) / 2
    U_min = min(U, nx_ * ny - U)
    mu = nx_ * ny / 2
    sigma = math.sqrt(nx_ * ny * (nx_ + ny + 1) / 12)
    if sigma == 0:
        return U_min, 1.0
    z = (U_min - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return U_min, p


print("=" * 95)
print("СЕРИЯ 6 (ПЕРЕРАБОТАННАЯ): УСТОЙЧИВОСТЬ НА НЕСКОЛЬКИХ ГРАФАХ")
print(f"n = {N_NODES}, p = {P_EDGE}, графов = {GRAPHS}, запусков на граф = {RUNS}")
print("=" * 95)

per_graph = []

for g_idx in range(GRAPHS):
    G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED + g_idx)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"\n{'=' * 70}")
    print(f"ГРАФ #{g_idx + 1}: {G.number_of_nodes()} вершин, "
          f"{G.number_of_edges()} рёбер, ср.степень "
          f"{2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"{'=' * 70}")

    ga_lens, sa_lens = [], []

    for r in range(RUNS):
        ga = GeneticAlgorithm(G)
        p = ga.solve()
        ga_lens.append(len(p) - 1)

        sa = SimulatedAnnealing(G)
        p = sa.solve()
        sa_lens.append(len(p) - 1)

        if (r + 1) % 5 == 0:
            print(f"  ...прогон {r + 1}/{RUNS}: "
                  f"ЭГА посл.={ga_lens[-1]}, ИО посл.={sa_lens[-1]}")

    ga_mean = statistics.mean(ga_lens)
    ga_std = statistics.stdev(ga_lens)
    ga_ci = confidence_interval_95(ga_lens)
    sa_mean = statistics.mean(sa_lens)
    sa_std = statistics.stdev(sa_lens)
    sa_ci = confidence_interval_95(sa_lens)
    U, pval = mann_whitney(ga_lens, sa_lens)

    print(f"\n  ЭГА: среднее={ga_mean:.2f}, std={ga_std:.2f}, "
          f"min/max={min(ga_lens)}/{max(ga_lens)}, "
          f"95% ДИ=[{ga_ci[0]:.2f}; {ga_ci[1]:.2f}]")
    print(f"  ИО:  среднее={sa_mean:.2f}, std={sa_std:.2f}, "
          f"min/max={min(sa_lens)}/{max(sa_lens)}, "
          f"95% ДИ=[{sa_ci[0]:.2f}; {sa_ci[1]:.2f}]")
    print(f"  Разрыв (ИО - ЭГА): {sa_mean - ga_mean:+.2f}")
    print(f"  U Манна-Уитни = {U:.1f}, p-value ~ {pval:.5f} "
          f"({'значимо' if pval < 0.05 else 'НЕ значимо'})")

    per_graph.append({
        "graph": g_idx + 1, "edges": G.number_of_edges(),
        "ga_mean": ga_mean, "ga_std": ga_std, "ga_min": min(ga_lens),
        "ga_max": max(ga_lens), "ga_ci": ga_ci,
        "sa_mean": sa_mean, "sa_std": sa_std, "sa_min": min(sa_lens),
        "sa_max": max(sa_lens), "sa_ci": sa_ci,
        "gap": sa_mean - ga_mean, "U": U, "pval": pval,
    })

# ============ СВОДНАЯ ТАБЛИЦА ПО ГРАФАМ ============
print("\n" + "=" * 110)
print("СВОДНАЯ ТАБЛИЦА ПО ГРАФАМ (новая Таблица 2.7)")
print("=" * 110)
print(f"{'Граф':>5} | {'Рёбер':>6} | {'ЭГА ср':>7} | {'ЭГА s':>6} | "
      f"{'ИО ср':>7} | {'ИО s':>6} | {'Разрыв':>7} | {'p-value':>9}")
print("-" * 110)
for r in per_graph:
    print(f"{r['graph']:>5} | {r['edges']:>6} | {r['ga_mean']:>7.2f} | "
          f"{r['ga_std']:>6.2f} | {r['sa_mean']:>7.2f} | {r['sa_std']:>6.2f} | "
          f"{r['gap']:>+7.2f} | {r['pval']:>9.5f}")
print("-" * 110)

all_ga_means = [r["ga_mean"] for r in per_graph]
all_sa_means = [r["sa_mean"] for r in per_graph]
all_gaps = [r["gap"] for r in per_graph]
all_ga_std = [r["ga_std"] for r in per_graph]
all_sa_std = [r["sa_std"] for r in per_graph]

print("\nАГРЕГИРОВАННЫЕ ПОКАЗАТЕЛИ ПО ВСЕМ ГРАФАМ:")
print(f"  ЭГА: среднее средних = {statistics.mean(all_ga_means):.2f}, "
      f"средний std внутри графа = {statistics.mean(all_ga_std):.2f}")
print(f"  ИО:  среднее средних = {statistics.mean(all_sa_means):.2f}, "
      f"средний std внутри графа = {statistics.mean(all_sa_std):.2f}")
print(f"  Разрыв (ИО - ЭГА): среднее = {statistics.mean(all_gaps):+.2f}, "
      f"min = {min(all_gaps):+.2f}, max = {max(all_gaps):+.2f}")
n_sa_wins = sum(1 for g in all_gaps if g > 0.5)
n_ties = sum(1 for g in all_gaps if abs(g) <= 0.5)
n_ga_wins = sum(1 for g in all_gaps if g < -0.5)
print(f"  Графов, где лидирует ИО: {n_sa_wins}/{GRAPHS}, "
      f"ничьих: {n_ties}/{GRAPHS}, лидирует ЭГА: {n_ga_wins}/{GRAPHS}")
print("\nЕсли разброс колонки 'Разрыв' узкий и знак один и тот же,")
print("эффект общий, а не особенность конкретного графа.")
print("=" * 110)
