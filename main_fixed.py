import networkx as nx
import random
import math
import time


class LongestPathSolver:
    """Базовый класс с общими утилитами."""

    def __init__(self, graph):
        self.graph = graph
        self.best_path = []
        self.history = []
        self.fitness_evaluations = 0  # счётчик NCE для серии 7

    def is_valid_path(self, path):
        """Жёсткая проверка валидности: все рёбра существуют, дубликатов нет."""
        if not path or len(path) < 2:
            return True
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False
        return True

    def greedy_path(self, start_node=None):
        """Жадное наращивание пути по соседу с максимальной степенью."""
        if not self.graph.nodes():
            return []
        if start_node is None:
            start_node = random.choice(list(self.graph.nodes()))
        path = [start_node]
        visited = {start_node}
        while True:
            current = path[-1]
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                break
            next_node = max(neighbors, key=lambda n: self.graph.degree(n))
            path.append(next_node)
            visited.add(next_node)
        return path

    def random_walk(self, start_node=None, max_steps=100):
        """Случайное блуждание без повторов."""
        if not self.graph.nodes():
            return []
        if start_node is None:
            start_node = random.choice(list(self.graph.nodes()))
        path = [start_node]
        visited = {start_node}
        steps = 0
        while steps < max_steps:
            current = path[-1]
            neighbors = [n for n in self.graph.neighbors(current) if n not in visited]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited.add(next_node)
            steps += 1
        return path


class GeneticAlgorithm(LongestPathSolver):
    """Элитарный ГА с кроссовером PPX (Path-Preserving Crossover)."""

    def __init__(self, graph, pop_size=100, generations=500, elite_size=10,
                 tournament_k=5, p_crossover=0.9, p_mutation=0.6,
                 stagnation_limit=80):
        super().__init__(graph)
        self.pop_size = pop_size
        self.generations = generations
        self.elite_size = elite_size
        self.tournament_k = tournament_k
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.stagnation_limit = stagnation_limit

    # ---------- инициализация ----------
    def initialize_population(self):
        """Гибридная инициализация: 20% жадные, 60% RW длинные, 20% RW короткие."""
        population = []
        nodes = list(self.graph.nodes())
        if not nodes:
            return []

        target_greedy = int(self.pop_size * 0.2)
        target_long_rw = int(self.pop_size * 0.6)

        for _ in range(target_greedy):
            path = self.greedy_path(random.choice(nodes))
            if self.is_valid_path(path):
                population.append(path)

        for _ in range(target_long_rw):
            path = self.random_walk(random.choice(nodes),
                                    max_steps=random.randint(10, 80))
            if self.is_valid_path(path):
                population.append(path)

        while len(population) < self.pop_size:
            path = self.random_walk(random.choice(nodes),
                                    max_steps=random.randint(3, 15))
            if self.is_valid_path(path) and len(path) >= 1:
                population.append(path)
            else:
                population.append([random.choice(nodes)])
        return population

    # ---------- селекция ----------
    def fitness(self, path):
        self.fitness_evaluations += 1
        if not self.is_valid_path(path):
            return -1000
        return len(path) - 1

    def tournament_selection(self, population):
        tournament = random.sample(population, min(self.tournament_k, len(population)))
        return max(tournament, key=self.fitness)

    # ---------- кроссовер PPX ----------
    def ppx_crossover(self, parent1, parent2):
        """
        Path-Preserving Crossover.
        Идея: найти общую вершину родителей, склеить префикс P1 до неё с суффиксом P2 после неё.
        Все рёбра потомка наследуются от одного из родителей — нет искусственных связей.
        """
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy()

        common = list(set(parent1) & set(parent2))
        if not common:
            # нет точки склейки — возвращаем родителя без изменений
            return parent1.copy()

        # выбираем точку склейки случайно среди общих вершин
        pivot = random.choice(common)
        idx1 = parent1.index(pivot)
        idx2 = parent2.index(pivot)

        prefix = parent1[:idx1 + 1]            # включая pivot
        suffix = parent2[idx2 + 1:]            # после pivot
        candidate = prefix + suffix

        # обрезаем по первому повтору вершины (если возник цикл при склейке)
        seen = set()
        truncated = []
        for v in candidate:
            if v in seen:
                break
            truncated.append(v)
            seen.add(v)

        if self.is_valid_path(truncated) and len(truncated) >= 2:
            return truncated
        return parent1.copy()

    # ---------- мутации ----------
    def mutate(self, path):
        if len(path) < 2:
            return path
        new_path = path.copy()
        roll = random.random()

        # 1. Расширение пути (40%)
        if roll < 0.4:
            current_end = new_path[-1]
            free_nbrs = [n for n in self.graph.neighbors(current_end) if n not in new_path]
            steps = random.randint(1, 3)
            while free_nbrs and steps > 0 and len(new_path) < self.graph.number_of_nodes():
                nxt = random.choice(free_nbrs)
                if self.graph.has_edge(new_path[-1], nxt):
                    new_path.append(nxt)
                    free_nbrs = [n for n in self.graph.neighbors(nxt) if n not in new_path]
                    steps -= 1
                else:
                    break

        # 2. Инверсия подпути (30%)
        elif roll < 0.7 and len(new_path) >= 3:
            i, j = sorted(random.sample(range(len(new_path)), 2))
            if j - i > 1:
                segment = new_path[i:j + 1]
                segment.reverse()
                new_path[i:j + 1] = segment
                if not self.is_valid_path(new_path):
                    new_path = path.copy()  # откат

        # 3. Вырезание сегмента (30%)
        elif len(new_path) >= 4:
            i = random.randint(1, len(new_path) - 3)
            j = random.randint(i + 1, len(new_path) - 2)
            if self.graph.has_edge(new_path[i - 1], new_path[j + 1]):
                new_path = new_path[:i] + new_path[j + 1:]

        return new_path if self.is_valid_path(new_path) else path

    # ---------- основной цикл ----------
    def solve(self, log_callback=None):
        self.fitness_evaluations = 0
        population = self.initialize_population()
        self.history = []
        best_overall = -1
        stagnation = 0

        for gen in range(self.generations):
            population = [p for p in population if self.is_valid_path(p)]
            if not population:
                population = self.initialize_population()

            population.sort(key=self.fitness, reverse=True)
            best_fitness = self.fitness(population[0])
            self.history.append(best_fitness)

            if best_fitness > best_overall:
                best_overall = best_fitness
                stagnation = 0
            else:
                stagnation += 1

            if log_callback and gen % 50 == 0:
                log_callback(f"Gen {gen}: best={best_fitness}, stagn={stagnation}")

            # ранний останов
            if stagnation >= self.stagnation_limit:
                if log_callback:
                    log_callback(f"Ранний останов на поколении {gen}")
                break

            # элитизм
            new_pop = population[:self.elite_size]

            # порождение потомков
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


class SimulatedAnnealing(LongestPathSolver):
    """Метод имитации отжига с тремя типами ходов."""

    def __init__(self, graph, initial_temp=2000, cooling_rate=0.97,
                 iterations_per_temp=100, no_improve_limit=50, min_temp=0.01):
        super().__init__(graph)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations_per_temp = iterations_per_temp
        self.no_improve_limit = no_improve_limit
        self.min_temp = min_temp

    def get_neighbor(self, path):
        """
        Генерация соседнего решения. Четыре типа ходов:
          1) расширение в конец (40%);
          2) аккуратное удаление с конца или из середины с shortcut (25%);
          3) инверсия подпути / 2-opt (20%);
          4) вставка обходной вершины — удлинение (15%).
        Любой ход, нарушающий валидность, откатывается.
        """
        if not path or len(path) < 1:
            return path
        action = random.random()
        new_path = path.copy()

        # 1. Расширение в конец (40%) — главный ход роста
        if action < 0.4:
            end = new_path[-1]
            free_nbrs = [n for n in self.graph.neighbors(end) if n not in new_path]
            if free_nbrs:
                new_path.append(random.choice(free_nbrs))

        # 2. Удаление (25%) — но безопасное
        elif action < 0.65 and len(new_path) > 1:
            # Половина случаев — удаление с конца (всегда валидно).
            # Половина случаев — удаление из середины ТОЛЬКО если соседи
            # удаляемой вершины напрямую соединены ребром (shortcut).
            if random.random() < 0.5 or len(new_path) < 4:
                new_path.pop()
            else:
                idx = random.randint(1, len(new_path) - 2)
                if self.graph.has_edge(new_path[idx - 1], new_path[idx + 1]):
                    new_path.pop(idx)
                # иначе ход не делаем — путь остаётся как был

        # 3. Инверсия подпути / 2-opt (20%) — переупорядочивание без потери вершин
        elif action < 0.85 and len(new_path) >= 4:
            i, j = sorted(random.sample(range(len(new_path)), 2))
            if j - i >= 2:
                segment = new_path[i:j + 1]
                segment.reverse()
                new_path[i:j + 1] = segment
                if not self.is_valid_path(new_path):
                    new_path = path.copy()  # откат

        # 4. Вставка обходной вершины (15%) — удлинение пути
        else:
            if len(new_path) >= 2:
                i = random.randint(0, len(new_path) - 2)
                for k in self.graph.neighbors(new_path[i]):
                    if (k not in new_path
                            and self.graph.has_edge(k, new_path[i + 1])):
                        new_path.insert(i + 1, k)
                        break
        return new_path if self.is_valid_path(new_path) else path

    def length(self, path):
        self.fitness_evaluations += 1
        return len(path) - 1 if self.is_valid_path(path) else -1000

    def solve(self, log_callback=None):
        self.fitness_evaluations = 0
        current_path = self.greedy_path()
        current_len = self.length(current_path)
        best_path = current_path.copy()
        best_len = current_len

        temp = self.initial_temp
        no_improve = 0
        self.history = []

        while temp > self.min_temp and no_improve < self.no_improve_limit:
            improved = False
            for _ in range(self.iterations_per_temp):
                new_path = self.get_neighbor(current_path)
                new_len = self.length(new_path)
                delta = new_len - current_len
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_path = new_path
                    current_len = new_len
                    if current_len > best_len:
                        best_path = current_path.copy()
                        best_len = current_len
                        improved = True
                        no_improve = 0
            self.history.append(best_len)
            if log_callback and len(self.history) % 10 == 0:
                log_callback(f"T={temp:.2f}: best={best_len}")
            if not improved:
                no_improve += 1
            temp *= self.cooling_rate

        self.best_path = best_path
        return best_path


# ---------------- GUI ----------------
class LongestPathGUI:
    def __init__(self, root):
        # импорты GUI-зависимостей
        import tkinter as tk
        from tkinter import ttk, scrolledtext, messagebox
        import matplotlib.pyplot as plt
        self._tk = tk
        self._ttk = ttk
        self._scrolledtext = scrolledtext
        self._messagebox = messagebox
        self._plt = plt

        self.root = root
        self.root.title("Longest Path: ЭГА (PPX) vs ИО")
        self.root.geometry("1200x800")
        self.graph = None
        self._build()

    def _build(self):
        tk = self._tk
        ttk = self._ttk
        scrolledtext = self._scrolledtext

        top = ttk.Frame(self.root, padding=10)
        top.grid(row=0, column=0, columnspan=2, sticky="we")

        ttk.Label(top, text="Вершин:").grid(row=0, column=0, padx=5)
        self.n_var = tk.IntVar(value=50)
        ttk.Entry(top, textvariable=self.n_var, width=8).grid(row=0, column=1)

        ttk.Label(top, text="p:").grid(row=0, column=2, padx=5)
        self.p_var = tk.DoubleVar(value=0.3)
        ttk.Entry(top, textvariable=self.p_var, width=8).grid(row=0, column=3)

        ttk.Button(top, text="Сгенерировать", command=self.generate).grid(row=0, column=4, padx=5)
        ttk.Button(top, text="ЭГА", command=self.run_ga).grid(row=0, column=5, padx=5)
        ttk.Button(top, text="ИО", command=self.run_sa).grid(row=0, column=6, padx=5)
        ttk.Button(top, text="Сравнить", command=self.compare).grid(row=0, column=7, padx=5)

        left = ttk.Frame(self.root, padding=10)
        left.grid(row=1, column=0, sticky="nswe")
        self.fig, self.ax = self._plt.subplots(figsize=(6, 6))
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        right = ttk.Frame(self.root, padding=10)
        right.grid(row=1, column=1, sticky="nswe")
        ttk.Label(right, text="Лог:", font=("Arial", 12, "bold")).pack()
        self.log_text = scrolledtext.ScrolledText(right, width=50, height=30)
        self.log_text.pack(fill="both", expand=True)

        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

    def log(self, msg):
        self.log_text.insert(self._tk.END, msg + "\n")
        self.log_text.see(self._tk.END)
        self.root.update()

    def generate(self):
        n, p = self.n_var.get(), self.p_var.get()
        self.graph = nx.gnp_random_graph(n, p, directed=False)
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        self.log(f"Граф: {self.graph.number_of_nodes()} вершин, "
                 f"{self.graph.number_of_edges()} рёбер")
        self.draw()

    def draw(self, path=None):
        self.ax.clear()
        if not self.graph:
            return
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, ax=self.ax)
        nx.draw_networkx_nodes(self.graph, pos, node_color="lightblue",
                               node_size=400, ax=self.ax)
        nx.draw_networkx_labels(self.graph, pos, ax=self.ax, font_size=8)
        if path and len(path) > 1:
            edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges,
                                   edge_color="red", width=3, ax=self.ax)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=path,
                                   node_color="yellow", node_size=400, ax=self.ax)
        self.ax.set_title(f"|V|={self.graph.number_of_nodes()}")
        self.canvas.draw()

    def _run(self, solver, name):
        if not self.graph:
            self._messagebox.showwarning("!", "Сначала создайте граф")
            return None
        self.log(f"\n=== {name} ===")
        t0 = time.perf_counter()
        path = solver.solve(log_callback=self.log)
        dt = time.perf_counter() - t0
        self.log(f"{name}: |P|={len(path) - 1}, NCE={solver.fitness_evaluations}, t={dt:.3f}с")
        self.draw(path)
        return path, dt, solver.fitness_evaluations

    def run_ga(self):
        self._run(GeneticAlgorithm(self.graph), "ЭГА")

    def run_sa(self):
        self._run(SimulatedAnnealing(self.graph), "ИО")

    def compare(self):
        if not self.graph:
            self._messagebox.showwarning("!", "Сначала создайте граф")
            return
        self.log("\n=== Сравнение ===")
        ga = GeneticAlgorithm(self.graph)
        t0 = time.perf_counter(); ga_path = ga.solve(); ga_t = time.perf_counter() - t0
        sa = SimulatedAnnealing(self.graph)
        t0 = time.perf_counter(); sa_path = sa.solve(); sa_t = time.perf_counter() - t0

        self.log(f"ЭГА: |P|={len(ga_path) - 1}, NCE={ga.fitness_evaluations}, t={ga_t:.3f}с")
        self.log(f"ИО:  |P|={len(sa_path) - 1}, NCE={sa.fitness_evaluations}, t={sa_t:.3f}с")
        better = ga_path if len(ga_path) >= len(sa_path) else sa_path
        self.draw(better)


if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = LongestPathGUI(root)
    root.mainloop()
