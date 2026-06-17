"""
ЭКСПЕРИМЕНТ 5. Структурная зависимость: разные классы графов.
Тестируются: G(n,p), Барабаши-Альберта (безмасштабный), 3-регулярный,
DAG (ориентированный ациклический), сеточный.
Размер n = 100. На каждом классе — несколько графов и прогонов.
Запуск: python exp5_structure.py
"""

import time
import statistics
import networkx as nx
import random

try:
    from main_fixed import GeneticAlgorithm, SimulatedAnnealing
except ImportError:
    from main import GeneticAlgorithm, SimulatedAnnealing

# ----- параметры -----
N_NODES = 100
GRAPHS_PER_TYPE = 3
RUNS_PER_GRAPH = 3
SEED = 42

random.seed(SEED)


# ----- генераторы графов -----
def gen_erdos(n, seed):
    G = nx.gnp_random_graph(n, 0.2, seed=seed)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G

def gen_barabasi(n, seed):
    G = nx.barabasi_albert_graph(n, 3, seed=seed)
    return G

def gen_regular(n, seed):
    # 3-регулярный граф (n должно быть чётным для k=3 не подходит, нужно k*n чётно)
    G = nx.random_regular_graph(3, n if n % 2 == 0 else n + 1, seed=seed)
    return G

def gen_dag(n, seed):
    # DAG строим вручную: топологический порядок 0..n-1, рёбра только вперёд
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    edges_added = 0
    target = int(n * (n - 1) / 2 * 0.05)  # плотность 5% от max возможной
    while edges_added < target:
        u, v = rng.sample(range(n), 2)
        if u < v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    return G.to_undirected()  # для совместимости с алгоритмами

def gen_grid(n, seed):
    # квадратная сетка ~sqrt(n) x sqrt(n)
    side = int(n ** 0.5)
    G = nx.grid_2d_graph(side, side)
    G = nx.convert_node_labels_to_integers(G)
    return G


GRAPH_TYPES = {
    "erdos":    ("Эрдёш-Реньи G(n, 0.2)",     gen_erdos),
    "barabasi": ("Барабаши-Альберт (m=3)",     gen_barabasi),
    "regular":  ("3-регулярный",                gen_regular),
    "dag":      ("DAG (плотность 5%)",          gen_dag),
    "grid":     ("Сеточный 10x10",              gen_grid),
}

print("=" * 95)
print("СЕРИЯ 5: СТРУКТУРНАЯ ЗАВИСИМОСТЬ")
print(f"n = {N_NODES}, графов на тип = {GRAPHS_PER_TYPE}, прогонов на граф = {RUNS_PER_GRAPH}")
print("=" * 95)

results = {key: {"ga_len": [], "sa_len": [], "ga_t": [], "sa_t": [],
                 "graph_size": []} for key in GRAPH_TYPES}

for key, (descr, gen) in GRAPH_TYPES.items():
    print(f"\n--- {descr} ---")
    for g_idx in range(GRAPHS_PER_TYPE):
        G = gen(N_NODES, SEED + g_idx)
        size = G.number_of_nodes()
        edges = G.number_of_edges()
        results[key]["graph_size"].append(size)
        print(f"  Граф #{g_idx + 1}: {size} вершин, {edges} рёбер, "
              f"ср.степень={2 * edges / size:.2f}")

        for r in range(RUNS_PER_GRAPH):
            ga = GeneticAlgorithm(G)
            t0 = time.perf_counter()
            ga_p = ga.solve()
            ga_t = time.perf_counter() - t0
            ga_len = len(ga_p) - 1

            sa = SimulatedAnnealing(G)
            t0 = time.perf_counter()
            sa_p = sa.solve()
            sa_t = time.perf_counter() - t0
            sa_len = len(sa_p) - 1

            print(f"    Прогон {r + 1}: ЭГА |P|={ga_len:3d} ({ga_t:5.2f}с) | "
                  f"ИО |P|={sa_len:3d} ({sa_t:.3f}с)")
            results[key]["ga_len"].append(ga_len)
            results[key]["sa_len"].append(sa_len)
            results[key]["ga_t"].append(ga_t)
            results[key]["sa_t"].append(sa_t)

# ----- сводная таблица -----
print("\n" + "=" * 105)
print("СВОДНАЯ ТАБЛИЦА (Таблица 2.6 в отчёте)")
print("=" * 105)
print(f"{'Класс графа':<28} | {'|V|':>5} | {'ЭГА |P|':>8} | {'ИО |P|':>7} | "
      f"{'ЭГА σ':>6} | {'ИО σ':>5} | {'Лидер':>8}")
print("-" * 105)

for key, (descr, _) in GRAPH_TYPES.items():
    r = results[key]
    avg_size = statistics.mean(r["graph_size"])
    ga_avg = statistics.mean(r["ga_len"])
    sa_avg = statistics.mean(r["sa_len"])
    ga_std = statistics.stdev(r["ga_len"]) if len(r["ga_len"]) > 1 else 0.0
    sa_std = statistics.stdev(r["sa_len"]) if len(r["sa_len"]) > 1 else 0.0

    if abs(ga_avg - sa_avg) < 0.5:
        leader = "ничья"
    elif ga_avg > sa_avg:
        leader = "ЭГА"
    else:
        leader = "ИО"

    print(f"{descr:<28} | {avg_size:>5.0f} | {ga_avg:>8.1f} | {sa_avg:>7.1f} | "
          f"{ga_std:>6.2f} | {sa_std:>5.2f} | {leader:>8}")

print("=" * 105)
print("\nКомментарий: для DAG значение, близкое к |V|-1, означает, что найден")
print("путь длины почти равной диаметру — что для ациклического графа естественно.")
