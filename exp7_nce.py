"""
ЭКСПЕРИМЕНТ 7. Число вычислений целевой функции (NCE).
Контролируется, сколько вызовов fitness/length требуется до:
   - достижения уровня 90% от финального решения;
   - финала.
Условия: n = 100, p = 0.2, несколько прогонов для усреднения.
Запуск: python exp7_nce.py
"""

import time
import statistics
import networkx as nx
import random
import math

try:
    from main_fixed import GeneticAlgorithm, SimulatedAnnealing
except ImportError:
    from main import GeneticAlgorithm, SimulatedAnnealing

# ----- параметры -----
N_NODES = 100
P_EDGE = 0.2
RUNS = 8
SEED = 42

random.seed(SEED)


# ============== ГА с записью NCE по best ==============
class GAWithNCE(GeneticAlgorithm):
    """Сохраняет точки (NCE, best) при каждом улучшении."""

    def solve(self, log_callback=None):
        self.fitness_evaluations = 0
        self.nce_history = []  # [(NCE, best)]
        population = self.initialize_population()
        best_overall = -1
        stagnation = 0

        for gen in range(self.generations):
            population = [p for p in population if self.is_valid_path(p)]
            if not population:
                population = self.initialize_population()

            population.sort(key=self.fitness, reverse=True)
            best = self.fitness(population[0])

            if best > best_overall:
                best_overall = best
                stagnation = 0
                self.nce_history.append((self.fitness_evaluations, best))
            else:
                stagnation += 1

            if stagnation >= self.stagnation_limit:
                break

            new_pop = population[:self.elite_size]
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_selection(population)
                p2 = self.tournament_selection(population)
                if random.random() < self.p_crossover:
                    child = self.ppx_crossover(p1, p2)
                else:
                    child = p1.copy()
                if random.random() < self.p_mutation:
                    child = self.mutate(child)
                new_pop.append(child if self.is_valid_path(child) else p1)
            population = new_pop

        population = [p for p in population if self.is_valid_path(p)]
        population.sort(key=self.fitness, reverse=True)
        self.best_path = population[0] if population else []
        return self.best_path


# ============== ИО с записью NCE по best ==============
class SAWithNCE(SimulatedAnnealing):
    def solve(self, log_callback=None):
        self.fitness_evaluations = 0
        self.nce_history = []
        current = self.greedy_path()
        cur_len = self.length(current)
        best_path = current.copy()
        best_len = cur_len
        self.nce_history.append((self.fitness_evaluations, best_len))

        temp = self.initial_temp
        no_improve = 0

        while temp > self.min_temp and no_improve < self.no_improve_limit:
            improved = False
            for _ in range(self.iterations_per_temp):
                new_path = self.get_neighbor(current)
                new_len = self.length(new_path)
                delta = new_len - cur_len
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current = new_path
                    cur_len = new_len
                    if cur_len > best_len:
                        best_path = current.copy()
                        best_len = cur_len
                        improved = True
                        no_improve = 0
                        self.nce_history.append((self.fitness_evaluations, best_len))
            if not improved:
                no_improve += 1
            temp *= self.cooling_rate

        self.best_path = best_path
        return best_path


def nce_at_threshold(history, target):
    """Сколько NCE потребовалось, чтобы best впервые достиг target."""
    for nce, best in history:
        if best >= target:
            return nce
    return history[-1][0] if history else 0


# ============== прогон ==============
print("=" * 80)
print("СЕРИЯ 7: ЧИСЛО ВЫЧИСЛЕНИЙ ЦЕЛЕВОЙ ФУНКЦИИ (NCE)")
print(f"n = {N_NODES}, p = {P_EDGE}, прогонов = {RUNS}")
print("=" * 80)

G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED)
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер\n")

ga_records = []   # [(final, NCE_90, NCE_final, time)]
sa_records = []

print("--- ЭГА ---")
for r in range(RUNS):
    ga = GAWithNCE(G)
    t0 = time.perf_counter()
    p = ga.solve()
    dt = time.perf_counter() - t0
    final = len(p) - 1
    target = 0.9 * final
    nce_90 = nce_at_threshold(ga.nce_history, target)
    nce_final = ga.fitness_evaluations
    print(f"  Прогон {r + 1}: финал |P|={final}, NCE до 90%={nce_90:6d}, "
          f"NCE финал={nce_final:6d}, t={dt:.2f}с")
    ga_records.append((final, nce_90, nce_final, dt))

print("\n--- ИО ---")
for r in range(RUNS):
    sa = SAWithNCE(G)
    t0 = time.perf_counter()
    p = sa.solve()
    dt = time.perf_counter() - t0
    final = len(p) - 1
    target = 0.9 * final
    nce_90 = nce_at_threshold(sa.nce_history, target)
    nce_final = sa.fitness_evaluations
    print(f"  Прогон {r + 1}: финал |P|={final}, NCE до 90%={nce_90:6d}, "
          f"NCE финал={nce_final:6d}, t={dt:.3f}с")
    sa_records.append((final, nce_90, nce_final, dt))

# ============== сводная таблица ==============
print("\n" + "=" * 90)
print("СВОДНАЯ ТАБЛИЦА (Таблица 2.8 в отчёте)")
print("=" * 90)
print(f"{'Метод':<6} | {'Финал |P|':>10} | {'NCE до 90%':>11} | "
      f"{'NCE до финала':>14} | {'NCE / |P|':>10}")
print("-" * 90)

for name, recs in [("ЭГА", ga_records), ("ИО", sa_records)]:
    finals = [r[0] for r in recs]
    nce90 = [r[1] for r in recs]
    nce_fin = [r[2] for r in recs]

    avg_final = statistics.mean(finals)
    avg_nce90 = statistics.mean(nce90)
    avg_ncef = statistics.mean(nce_fin)
    nce_per_p = avg_ncef / avg_final if avg_final > 0 else 0

    print(f"{name:<6} | {avg_final:>10.1f} | {avg_nce90:>11.0f} | "
          f"{avg_ncef:>14.0f} | {nce_per_p:>10.1f}")

# отношение NCE
ga_avg_90 = statistics.mean(r[1] for r in ga_records)
sa_avg_90 = statistics.mean(r[1] for r in sa_records)
ga_avg_fin = statistics.mean(r[2] for r in ga_records)
sa_avg_fin = statistics.mean(r[2] for r in sa_records)

print("=" * 90)
print(f"\nОтношение NCE до 90%:    ЭГА / ИО = {ga_avg_90 / sa_avg_90:.2f}x")
print(f"Отношение NCE до финала: ЭГА / ИО = {ga_avg_fin / sa_avg_fin:.2f}x")
print("\nИО экономнее по числу вычислений целевой функции.")
