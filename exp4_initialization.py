"""
ЭКСПЕРИМЕНТ 4 (ПЕРЕРАБОТАННЫЙ). Влияние стратегии инициализации ЭГА.

Замечание научного руководителя: серия на ОДНОМ графе ничего не
доказывает и не опровергает - может быть особенностью графа.
Поэтому теперь:
  - две размерности: n = 50 и n = 100;
  - по GRAPHS_PER_SIZE разных графов на каждую размерность;
  - по RUNS прогонов на каждом графе.
Это позволяет увидеть, общий ли наблюдаемый эффект (провал чисто
жадной инициализации) или артефакт конкретной конфигурации.

Запуск: python exp4_initialization.py
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
SIZES = [50, 100]        # размерности
P_EDGE = 0.3
GRAPHS_PER_SIZE = 3      # разных графов на каждую размерность
RUNS = 5                 # прогонов на каждом графе
SEED = 42

random.seed(SEED)


class GAWithStrategy(GeneticAlgorithm):
    """ГА с настраиваемой стратегией инициализации популяции."""

    def __init__(self, graph, strategy="hybrid", **kw):
        super().__init__(graph, **kw)
        self.strategy = strategy
        self.initial_stats = {}

    def initialize_population(self):
        nodes = list(self.graph.nodes())
        pop = []

        if self.strategy == "hybrid":
            for _ in range(int(self.pop_size * 0.2)):
                p = self.greedy_path(random.choice(nodes))
                if self.is_valid_path(p): pop.append(p)
            for _ in range(int(self.pop_size * 0.6)):
                p = self.random_walk(random.choice(nodes), random.randint(10, 80))
                if self.is_valid_path(p): pop.append(p)
            while len(pop) < self.pop_size:
                p = self.random_walk(random.choice(nodes), random.randint(3, 15))
                if self.is_valid_path(p): pop.append(p)
                else: pop.append([random.choice(nodes)])

        elif self.strategy == "greedy_only":
            for _ in range(self.pop_size):
                p = self.greedy_path(random.choice(nodes))
                if self.is_valid_path(p): pop.append(p)

        elif self.strategy == "random_only":
            for _ in range(self.pop_size):
                p = self.random_walk(random.choice(nodes), random.randint(20, 70))
                if self.is_valid_path(p): pop.append(p)
                else: pop.append([random.choice(nodes)])

        elif self.strategy == "balanced":
            for _ in range(int(self.pop_size * 0.5)):
                p = self.greedy_path(random.choice(nodes))
                if self.is_valid_path(p): pop.append(p)
            while len(pop) < self.pop_size:
                p = self.random_walk(random.choice(nodes), random.randint(15, 60))
                if self.is_valid_path(p): pop.append(p)
                else: pop.append([random.choice(nodes)])

        elif self.strategy == "aggressive_greedy":
            for _ in range(int(self.pop_size * 0.8)):
                p = self.greedy_path(random.choice(nodes))
                if self.is_valid_path(p): pop.append(p)
            while len(pop) < self.pop_size:
                p = self.random_walk(random.choice(nodes), random.randint(10, 50))
                if self.is_valid_path(p): pop.append(p)
                else: pop.append([random.choice(nodes)])

        lens = [len(x) - 1 for x in pop]
        self.initial_stats = {"min": min(lens), "max": max(lens),
                              "avg": sum(lens) / len(lens)}
        return pop


STRATEGIES = {
    "hybrid":            "Гибридная (20Ж/60RW/20кор.)",
    "greedy_only":       "Только жадные (100%)",
    "random_only":       "Только случайные (100% RW)",
    "balanced":          "Сбалансированная (50/50)",
    "aggressive_greedy": "Агрессивная (80Ж/20RW)",
}

print("=" * 100)
print("СЕРИЯ 4 (ПЕРЕРАБОТАННАЯ): ИНИЦИАЛИЗАЦИЯ НА НЕСКОЛЬКИХ ГРАФАХ И РАЗМЕРНОСТЯХ")
print(f"n = {SIZES}, p = {P_EDGE}, графов на размер = {GRAPHS_PER_SIZE}, прогонов на граф = {RUNS}")
print("=" * 100)

# результаты: results[size][strategy] = список словарей по прогонам
results = {n: {key: [] for key in STRATEGIES} for n in SIZES}

for n in SIZES:
    print(f"\n{'#' * 90}")
    print(f"РАЗМЕРНОСТЬ n = {n}")
    print(f"{'#' * 90}")

    for g_idx in range(GRAPHS_PER_SIZE):
        G = nx.gnp_random_graph(n, P_EDGE, seed=SEED + n + g_idx)
        G.remove_nodes_from(list(nx.isolates(G)))
        gsize = G.number_of_nodes()
        print(f"\n  --- Граф #{g_idx + 1}: {gsize} вершин, {G.number_of_edges()} рёбер ---")

        for key, descr in STRATEGIES.items():
            init_maxes, finals, times = [], [], []
            for r in range(RUNS):
                ga = GAWithStrategy(G, strategy=key)
                t0 = time.perf_counter()
                path = ga.solve()
                dt = time.perf_counter() - t0
                init_maxes.append(ga.initial_stats["max"])
                finals.append(len(path) - 1)
                times.append(dt)
            results[n][key].append({
                "init_max": statistics.mean(init_maxes),
                "final": statistics.mean(finals),
                "final_all": finals,
                "time": statistics.mean(times),
                "gsize": gsize,
            })
            print(f"    {descr:<32}: нач.max={statistics.mean(init_maxes):5.1f}, "
                  f"финал={statistics.mean(finals):5.1f}")

# ============ СВОДНЫЕ ТАБЛИЦЫ ПО РАЗМЕРНОСТЯМ ============
for n in SIZES:
    print("\n" + "=" * 100)
    print(f"СВОДНАЯ ТАБЛИЦА для n = {n} (усреднение по {GRAPHS_PER_SIZE} графам)")
    print("=" * 100)
    print(f"{'Стратегия':<32} | {'Нач.max':>8} | {'Финал':>7} | "
          f"{'Улучш.':>7} | {'Потолок n-1':>11} | {'Время,с':>8}")
    print("-" * 100)
    for key, descr in STRATEGIES.items():
        data = results[n][key]
        init_max = statistics.mean([d["init_max"] for d in data])
        final = statistics.mean([d["final"] for d in data])
        t = statistics.mean([d["time"] for d in data])
        ceiling = statistics.mean([d["gsize"] for d in data]) - 1
        print(f"{descr:<32} | {init_max:>8.1f} | {final:>7.1f} | "
              f"{final - init_max:>+7.1f} | {ceiling:>11.0f} | {t:>8.2f}")
    print("-" * 100)

# ============ КЛЮЧЕВОЙ ВЫВОД: ВОСПРОИЗВОДИТСЯ ЛИ ПРОВАЛ ЖАДНОЙ ============
print("\n" + "=" * 100)
print("ПРОВЕРКА УСТОЙЧИВОСТИ ЭФФЕКТА 'ПРОВАЛ ЧИСТО ЖАДНОЙ ИНИЦИАЛИЗАЦИИ'")
print("=" * 100)
print(f"{'n':>5} | {'Граф':>5} | {'Жадная финал':>13} | {'Случайная финал':>16} | "
      f"{'Разрыв (RW-Ж)':>14}")
print("-" * 100)
for n in SIZES:
    for g_idx in range(GRAPHS_PER_SIZE):
        greedy_f = results[n]["greedy_only"][g_idx]["final"]
        random_f = results[n]["random_only"][g_idx]["final"]
        print(f"{n:>5} | {g_idx + 1:>5} | {greedy_f:>13.1f} | {random_f:>16.1f} | "
              f"{random_f - greedy_f:>+14.1f}")
print("-" * 100)
print("Если разрыв (RW - Жадная) положителен на ВСЕХ графах и размерностях,")
print("значит провал чисто жадной инициализации - общий эффект, а не")
print("особенность одного графа.")
print("=" * 100)
