"""
ЭКСПЕРИМЕНТ 3. Анализ сходимости.
Условия: n = 100, p = 0.15, единичный диагностический прогон.
Сохраняет историю лучшего и среднего фитнеса по поколениям/итерациям.
Строит график convergence.png и выводит ключевые точки в консоль.
Запуск: python exp3_convergence.py
"""

import time
import networkx as nx
import matplotlib.pyplot as plt
import random

try:
    from main_fixed import GeneticAlgorithm, SimulatedAnnealing, LongestPathSolver
except ImportError:
    from main import GeneticAlgorithm, SimulatedAnnealing, LongestPathSolver

# ----- параметры -----
N_NODES = 100
P_EDGE = 0.15
SEED = 236236263

random.seed(SEED)

print("=" * 80)
print("СЕРИЯ 3: АНАЛИЗ СХОДИМОСТИ")
print(f"n = {N_NODES}, p = {P_EDGE}")
print("=" * 80)

G = nx.gnp_random_graph(N_NODES, P_EDGE, seed=SEED)
G.remove_nodes_from(list(nx.isolates(G)))
print(f"Граф: {G.number_of_nodes()} вершин, {G.number_of_edges()} рёбер")
print(f"Средняя степень: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")


# ============== ГА с расширенным логом ==============
class GAWithFullHistory(GeneticAlgorithm):
    """ГА, сохраняющий best/avg/worst по поколениям."""

    def solve(self, log_callback=None):
        self.fitness_evaluations = 0
        population = self.initialize_population()
        self.history = []  # [{gen, best, avg, worst}]
        best_overall = -1
        stagnation = 0

        for gen in range(self.generations):
            population = [p for p in population if self.is_valid_path(p)]
            if not population:
                population = self.initialize_population()

            population.sort(key=self.fitness, reverse=True)
            fits = [self.fitness(p) for p in population]
            best, avg, worst = max(fits), sum(fits) / len(fits), min(fits)

            self.history.append({"gen": gen, "best": best, "avg": avg, "worst": worst})

            if best > best_overall:
                best_overall = best
                stagnation = 0
            else:
                stagnation += 1

            if gen % 50 == 0:
                print(f"  Gen {gen:3d}: best={best:3d}, avg={avg:6.1f}, worst={worst:3d}")

            if stagnation >= self.stagnation_limit:
                print(f"  Ранний останов на поколении {gen}")
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


# ============== ИО с расширенным логом ==============
class SAWithFullHistory(SimulatedAnnealing):
    """ИО, сохраняющий историю по итерациям."""

    def solve(self, log_callback=None):
        import math
        self.fitness_evaluations = 0
        current = self.greedy_path()
        cur_len = self.length(current)
        best_path = current.copy()
        best_len = cur_len

        print(f"  Стартовое решение (жадный): {cur_len}")

        temp = self.initial_temp
        no_improve = 0
        iteration = 0
        self.history = []  # [{iter, T, current, best}]

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
                if iteration % 20 == 0:
                    self.history.append({
                        "iter": iteration, "T": temp,
                        "current": cur_len, "best": best_len
                    })
                iteration += 1

            if not improved:
                no_improve += 1
            else:
                print(f"  T={temp:6.2f}: best={best_len}")
            temp *= self.cooling_rate

        self.best_path = best_path
        return best_path


# ============== прогон ==============
print("\n--- Запуск ЭГА ---")
ga = GAWithFullHistory(G)
t0 = time.perf_counter()
ga_path = ga.solve()
ga_t = time.perf_counter() - t0
ga_final = len(ga_path) - 1
print(f"ЭГА: финал |P|={ga_final}, поколений={len(ga.history)}, "
      f"NCE={ga.fitness_evaluations}, t={ga_t:.2f}с")

print("\n--- Запуск ИО ---")
sa = SAWithFullHistory(G)
t0 = time.perf_counter()
sa_path = sa.solve()
sa_t = time.perf_counter() - t0
sa_final = len(sa_path) - 1
print(f"ИО: финал |P|={sa_final}, итераций≈{sa.history[-1]['iter']}, "
      f"NCE={sa.fitness_evaluations}, t={sa_t:.2f}с")

# ============== ключевые точки сходимости ==============
print("\n" + "=" * 80)
print("КЛЮЧЕВЫЕ ТОЧКИ СХОДИМОСТИ ЭГА")
print("=" * 80)
checkpoints = [0, 50, 100, 200, 300, 400]
for cp in checkpoints:
    if cp < len(ga.history):
        h = ga.history[cp]
        print(f"  Gen {cp:3d}: best={h['best']:3d}, avg={h['avg']:6.1f}")
print(f"  Финал:    best={ga_final}")

print("\nКЛЮЧЕВЫЕ ТОЧКИ СХОДИМОСТИ ИО")
print("=" * 80)
sa_iters = sa.history
n_pts = len(sa_iters)
for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    idx = min(int(n_pts * frac), n_pts - 1)
    h = sa_iters[idx]
    print(f"  Iter {h['iter']:5d} (T={h['T']:7.2f}): "
          f"current={h['current']:3d}, best={h['best']:3d}")

# ============== итоговая сводка ==============
print("\n" + "=" * 80)
print("СВОДКА (для отчёта, серия 3)")
print("=" * 80)
print(f"  Старт ЭГА (best gen 0):  {ga.history[0]['best']}")
print(f"  Финал ЭГА:               {ga_final}  (улучшение +{ga_final - ga.history[0]['best']})")
print(f"  Старт ИО (жадный):       {sa.history[0]['best']}")
print(f"  Финал ИО:                {sa_final}  (улучшение +{sa_final - sa.history[0]['best']})")
print(f"  Время ЭГА: {ga_t:.2f}с | Время ИО: {sa_t:.3f}с | разрыв x{ga_t / sa_t:.0f}")

# ============== график ==============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
gens = [h["gen"] for h in ga.history]
best = [h["best"] for h in ga.history]
avg = [h["avg"] for h in ga.history]
ax.plot(gens, best, "b-", linewidth=2, label="ЭГА: лучшее")
ax.plot(gens, avg, "g--", linewidth=1.5, alpha=0.7, label="ЭГА: среднее")
ax.set_xlabel("Поколение")
ax.set_ylabel("Длина пути")
ax.set_title("Сходимость ЭГА")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
iters = [h["iter"] for h in sa.history]
sa_best = [h["best"] for h in sa.history]
sa_cur = [h["current"] for h in sa.history]
ax.plot(iters, sa_best, "r-", linewidth=2, label="ИО: лучшее")
ax.plot(iters, sa_cur, "orange", linewidth=1, alpha=0.5, label="ИО: текущее")
ax.set_xlabel("Итерация")
ax.set_ylabel("Длина пути")
ax.set_title("Сходимость ИО")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig("convergence.png", dpi=200, bbox_inches="tight")
print("\nГрафик сохранён: convergence.png")
plt.show()
