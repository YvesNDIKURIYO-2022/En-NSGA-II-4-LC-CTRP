import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import stats
import warnings
import os
from typing import Dict, List, Tuple, Any

warnings.filterwarnings('ignore')
os.makedirs('analysis_plots', exist_ok=True)
os.makedirs('analysis_plots/pareto_fronts', exist_ok=True)


class MultiObjectiveProblem:
    def __init__(self, n_var: int, n_obj: int, n_constr: int = 0,
                 xl: float = 0.0, xu: float = 1.0):
        self.n_var = n_var
        self.n_obj = n_obj
        self.n_constr = n_constr
        self.xl = xl
        self.xu = xu
        self.name = "BaseProblem"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        raise NotImplementedError

    def generate_population(self, size: int) -> np.ndarray:
        return np.random.random((size, self.n_var)) * (self.xu - self.xl) + self.xl

    def get_reference_point(self) -> np.ndarray:
        raise NotImplementedError

    def get_igd_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5]) if self.n_obj == 2 else np.array([2.0, 2.0, 2.0])


class ZDT1(MultiObjectiveProblem):
    def __init__(self, n_var: int = 10):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self.name = "ZDT1"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    def get_reference_point(self) -> np.ndarray:
        return np.array([11.0, 11.0])


class ZDT2(MultiObjectiveProblem):
    def __init__(self, n_var: int = 10):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self.name = "ZDT2"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])

    def get_reference_point(self) -> np.ndarray:
        return np.array([11.0, 11.0])


class ZDT3(MultiObjectiveProblem):
    def __init__(self, n_var: int = 10):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self.name = "ZDT3"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1 - np.sqrt(f1) - f1 * np.sin(10 * np.pi * f1)
        valid = f2 > 0
        return np.column_stack([f1[valid], f2[valid]])

    def get_reference_point(self) -> np.ndarray:
        return np.array([11.0, 11.0])


class ZDT4(MultiObjectiveProblem):
    def __init__(self, n_var: int = 5):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self.name = "ZDT4"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 10.0 * (self.n_var - 1) + np.sum(
            x[1:]**2 - 10.0 * np.cos(4 * np.pi * x[1:])
        )
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0, 1, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])

    def get_reference_point(self) -> np.ndarray:
        return np.array([100.0, 100.0])


class ZDT6(MultiObjectiveProblem):
    def __init__(self, n_var: int = 5):
        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0)
        self.name = "ZDT6"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f1 = 1.0 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0]) ** 6
        g = 1.0 + 9.0 * (np.sum(x[1:]) / (self.n_var - 1)) ** 0.25
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.280775, 1, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])

    def get_reference_point(self) -> np.ndarray:
        return np.array([11.0, 11.0])


class DTLZ1(MultiObjectiveProblem):
    def __init__(self, n_var: int = 5, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ1"
        self.k = self.n_var - self.n_obj + 1

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = 100.0 * (self.k + np.sum(
            (x[self.n_obj-1:] - 0.5) ** 2 -
            np.cos(20 * np.pi * (x[self.n_obj-1:] - 0.5))
        ))
        f = 0.5 * (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= x[j]
            if i > 0:
                f[i] *= (1 - x[self.n_obj - i - 1])
        return f

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.dirichlet(np.ones(self.n_obj), n_points)
        return 0.5 * points

    def get_reference_point(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0])


class DTLZ2(MultiObjectiveProblem):
    def __init__(self, n_var: int = 6, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ2"
        self.k = self.n_var - self.n_obj + 1

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = np.sum((x[self.n_obj-1:] - 0.5) ** 2)
        f = (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x[j] * np.pi / 2.0)
            if i > 0:
                f[i] *= np.sin(x[self.n_obj - i - 1] * np.pi / 2.0)
        return f

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sqrt(np.sum(points ** 2, axis=1, keepdims=True))
        return points

    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class DTLZ3(DTLZ2):
    def __init__(self, n_var: int = 6, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "DTLZ3"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = 100.0 * (self.k + np.sum(
            (x[self.n_obj-1:] - 0.5) ** 2 -
            np.cos(20 * np.pi * (x[self.n_obj-1:] - 0.5))
        ))
        f = (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x[j] * np.pi / 2.0)
            if i > 0:
                f[i] *= np.sin(x[self.n_obj - i - 1] * np.pi / 2.0)
        return f

    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class DTLZ4(DTLZ2):
    def __init__(self, n_var: int = 6, n_obj: int = 3, alpha: float = 10.0):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "DTLZ4"
        self.alpha = alpha

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_alpha = x[:self.n_obj-1] ** self.alpha
        g = np.sum((x[self.n_obj-1:] - 0.5) ** 2)
        f = (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(x_alpha[j] * np.pi / 2.0)
            if i > 0:
                f[i] *= np.sin(x_alpha[self.n_obj - i - 1] * np.pi / 2.0)
        return f

    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class WFG1(MultiObjectiveProblem):
    def __init__(self, n_var: int = 6, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "WFG1"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            start = i * (self.n_var // self.n_obj)
            end = start + (self.n_var // self.n_obj)
            f[i] = np.mean(x[start:end]) * (i + 1)
        return f

    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.rand(n_points, self.n_obj)
        points = points / np.sum(points, axis=1, keepdims=True)
        return points

    def get_reference_point(self) -> np.ndarray:
        return np.array([3.0, 3.0, 3.0])


class WFG2(WFG1):
    def __init__(self, n_var: int = 6, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG2"

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            f[i] = np.sum(x[i::self.n_obj] ** 2) / (self.n_var // self.n_obj)
        return f


class BaseMOEA:
    def __init__(self, pop_size: int = 50, max_gen: int = 100,
                 crossover_prob: float = 0.9, mutation_prob: float = 0.1,
                 eta_c: float = 20.0, eta_m: float = 20.0):

        self.pop_size = pop_size
        self.max_gen = max_gen
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.name = "BaseMOEA"

    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        raise NotImplementedError

    def fast_non_dominated_sort(self, F: np.ndarray) -> List[List[int]]:
        n = len(F)
        S = [[] for _ in range(n)]
        front = [[]]
        n_p = np.zeros(n, dtype=int)
        rank = np.zeros(n, dtype=int)

        for i in range(n):
            for j in range(n):
                if i != j:
                    if self.dominates(F[i], F[j]):
                        S[i].append(j)
                    elif self.dominates(F[j], F[i]):
                        n_p[i] += 1
            if n_p[i] == 0:
                rank[i] = 0
                front[0].append(i)

        i = 0
        while front[i]:
            Q = []
            for p in front[i]:
                for q in S[p]:
                    n_p[q] -= 1
                    if n_p[q] == 0:
                        rank[q] = i + 1
                        Q.append(q)
            i += 1
            front.append(Q)

        return front[:-1]

    def dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        return np.all(a <= b) and np.any(a < b)

    def crowding_distance(self, F: np.ndarray, front: List[int]) -> np.ndarray:
        n = len(front)
        m = F.shape[1]
        distance = np.zeros(n)

        if n <= 2:
            distance[:] = 1e6
            return distance

        for obj in range(m):
            sorted_indices = np.argsort(F[front, obj])
            sorted_front = [front[i] for i in sorted_indices]

            distance[sorted_indices[0]] = 1e6
            distance[sorted_indices[-1]] = 1e6

            f_min = F[sorted_front[0], obj]
            f_max = F[sorted_front[-1], obj]

            if abs(f_max - f_min) < 1e-10:
                continue

            for i in range(1, n - 1):
                idx = sorted_indices[i]
                distance[idx] += (F[sorted_front[i + 1], obj] -
                                  F[sorted_front[i - 1], obj]) / (f_max - f_min)

        return distance

    def simulated_binary_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_var = len(parent1)

        if np.random.random() < self.crossover_prob:
            u = np.random.random(n_var)
            beta = np.zeros(n_var)
            mask = u <= 0.5
            beta[mask] = (2.0 * u[mask]) ** (1.0 / (self.eta_c + 1.0))
            beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** (1.0 / (self.eta_c + 1.0))

            child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
            child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

            child1 = np.clip(child1, 0.0, 1.0)
            child2 = np.clip(child2, 0.0, 1.0)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        return child1, child2

    def polynomial_mutation(self, individual: np.ndarray) -> np.ndarray:
        mutant = individual.copy()
        n_var = len(individual)

        mutation_mask = np.random.random(n_var) < self.mutation_prob
        if np.any(mutation_mask):
            u = np.random.random(np.sum(mutation_mask))
            delta = np.zeros_like(u)

            mask1 = u < 0.5
            delta[mask1] = (2.0 * u[mask1]) ** (1.0 / (self.eta_m + 1.0)) - 1.0
            delta[~mask1] = 1.0 - (2.0 * (1.0 - u[~mask1])) ** (1.0 / (self.eta_m + 1.0))

            mutant[mutation_mask] += delta * 0.1
            mutant = np.clip(mutant, 0.0, 1.0)

        return mutant


class NSGA2(BaseMOEA):
    def __init__(self, pop_size: int = 50, max_gen: int = 100):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "NSGA-II"

    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()

        population = problem.generate_population(self.pop_size)
        objectives = np.array([problem.evaluate(ind) for ind in population])

        history = {'generation': [], 'best_hv': [], 'best_igd': [], 'avg_objectives': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)

        for gen in range(self.max_gen):
            offspring_pop = []

            for _ in range(self.pop_size // 2):
                candidates = np.random.choice(self.pop_size, 4, replace=False)
                winner1 = candidates[0]
                winner2 = candidates[2]

                for i in range(1, 2):
                    if self.dominates(objectives[candidates[i]], objectives[winner1]):
                        winner1 = candidates[i]

                for i in range(3, 4):
                    if self.dominates(objectives[candidates[i]], objectives[winner2]):
                        winner2 = candidates[i]

                parent1 = population[winner1]
                parent2 = population[winner2]

                child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                child1 = self.polynomial_mutation(child1)
                child2 = self.polynomial_mutation(child2)

                offspring_pop.extend([child1, child2])

            offspring_obj = np.array([problem.evaluate(ind) for ind in offspring_pop])

            combined_pop = np.vstack([population, offspring_pop])
            combined_obj = np.vstack([objectives, offspring_obj])

            fronts = self.fast_non_dominated_sort(combined_obj)

            new_pop = []
            new_obj = []
            front_idx = 0

            while len(new_pop) + len(fronts[front_idx]) <= self.pop_size:
                for idx in fronts[front_idx]:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
                front_idx += 1

            if len(new_pop) < self.pop_size:
                remaining = self.pop_size - len(new_pop)
                front_indices = fronts[front_idx]

                if len(front_indices) > 1:
                    crowding = self.crowding_distance(combined_obj, front_indices)
                    sorted_idx = np.argsort(crowding)[::-1][:remaining]
                    for idx in sorted_idx:
                        new_pop.append(combined_pop[front_indices[idx]])
                        new_obj.append(combined_obj[front_indices[idx]])
                else:
                    for idx in front_indices[:remaining]:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])

            population = np.array(new_pop)
            objectives = np.array(new_obj)

            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)

            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
            history['avg_objectives'].append(np.mean(objectives, axis=0))

        runtime = time.time() - start_time

        final_hv = metrics_calc.hypervolume(objectives, ref_point)
        final_igd = metrics_calc.inverted_generational_distance(objectives, true_front)

        return {
            'population': population,
            'objectives': objectives,
            'runtime': runtime,
            'history': history,
            'final_hv': final_hv,
            'final_igd': final_igd,
            'algorithm': self.name
        }


class EnhancedNSGA2(NSGA2):
    def __init__(self, pop_size: int = 50, max_gen: int = 100, use_enhancements: bool = True):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "En-NSGA-II"
        self.use_enhancements = use_enhancements
        self.archive_size = pop_size // 2
        self.archive = None
        self.archive_objectives = None
        self.adaptive_mutation_rate = self.mutation_prob
        self.adaptive_crossover_rate = self.crossover_prob
        self.diversity_threshold = 0.1
        self.restart_counter = 0
        self.restart_threshold = 20
        self.best_hv_history = []

    def intelligent_initialization(self, problem: MultiObjectiveProblem, size: int) -> np.ndarray:
        population = []
        n_var = problem.n_var

        lhs_points = np.zeros((size, n_var))
        for i in range(n_var):
            lhs_points[:, i] = np.random.permutation(size) / (size - 1)

        lhs_points = lhs_points * (problem.xu - problem.xl) + problem.xl
        population.extend(lhs_points)

        if len(population) < size:
            extra_needed = size - len(population)
            if problem.name.startswith('ZDT'):
                for i in range(extra_needed):
                    x = np.random.random(n_var)
                    x[0] = i / (extra_needed - 1) if extra_needed > 1 else 0.5
                    x[1:] = np.random.random(n_var - 1) * 0.2
                    population.append(x)
            else:
                population.extend(problem.generate_population(extra_needed))

        return np.array(population[:size])

    def adaptive_tournament_selection(self, population: np.ndarray, objectives: np.ndarray,
                                     n_candidates: int = 8) -> Tuple[int, int]:
        candidates = np.random.choice(len(population), n_candidates, replace=False)

        fitness_scores = []
        for idx in candidates:
            domination_count = 0
            for j in range(len(population)):
                if self.dominates(objectives[idx], objectives[j]):
                    domination_count += 1

            distances = np.sqrt(np.sum((objectives[candidates] - objectives[idx]) ** 2, axis=1))
            diversity = np.mean(distances)

            fitness = domination_count + diversity * 0.1
            fitness_scores.append(fitness)

        top_indices = np.argsort(fitness_scores)[-2:]
        return candidates[top_indices[0]], candidates[top_indices[1]]

    def differential_evolution_crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                                       population: np.ndarray) -> np.ndarray:
        F = 0.5
        CR = 0.9

        idxs = np.random.choice(len(population), 3, replace=False)
        x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]

        v = x1 + F * (x2 - x3)
        v = np.clip(v, 0.0, 1.0)

        trial = parent1.copy()
        for i in range(len(parent1)):
            if np.random.random() < CR or i == np.random.randint(len(parent1)):
                trial[i] = v[i]

        return trial

    def gaussian_mutation(self, individual: np.ndarray, generation: int, max_gen: int) -> np.ndarray:
        mutant = individual.copy()
        n_var = len(individual)

        adaptive_rate = self.adaptive_mutation_rate * (1.0 + generation / max_gen)
        sigma = 0.1 * (1.0 - generation / max_gen)

        for i in range(n_var):
            if np.random.random() < adaptive_rate:
                mutant[i] += np.random.normal(0, sigma)

        mutant = np.clip(mutant, 0.0, 1.0)
        return mutant

    def update_archive(self, population: np.ndarray, objectives: np.ndarray):
        if self.archive is None:
            self.archive = population.copy()
            self.archive_objectives = objectives.copy()
            return

        combined_pop = np.vstack([self.archive, population])
        combined_obj = np.vstack([self.archive_objectives, objectives])

        fronts = self.fast_non_dominated_sort(combined_obj)

        new_archive = []
        new_archive_obj = []
        front_idx = 0

        while len(new_archive) + len(fronts[front_idx]) <= self.archive_size:
            for idx in fronts[front_idx]:
                new_archive.append(combined_pop[idx])
                new_archive_obj.append(combined_obj[idx])
            front_idx += 1

        if len(new_archive) < self.archive_size:
            remaining = self.archive_size - len(new_archive)
            front_indices = fronts[front_idx]
            if len(front_indices) > 1:
                crowding = self.crowding_distance(combined_obj, front_indices)
                sorted_idx = np.argsort(crowding)[::-1][:remaining]
                for idx in sorted_idx:
                    new_archive.append(combined_pop[front_indices[idx]])
                    new_archive_obj.append(combined_obj[front_indices[idx]])

        self.archive = np.array(new_archive)
        self.archive_objectives = np.array(new_archive_obj)

    def inject_archive_solutions(self, population: np.ndarray, objectives: np.ndarray,
                               injection_rate: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        if self.archive is None or len(self.archive) == 0:
            return population, objectives

        n_inject = int(len(population) * injection_rate)
        if n_inject == 0:
            n_inject = 1

        if len(self.archive) > n_inject:
            distances = np.zeros(len(self.archive))
            for i in range(len(self.archive)):
                for j in range(len(self.archive)):
                    if i != j:
                        distances[i] += np.sqrt(np.sum((self.archive_objectives[i] -
                                                       self.archive_objectives[j]) ** 2))

            selected_idx = np.argsort(distances)[-n_inject:]
            archive_selected = self.archive[selected_idx]
            archive_obj_selected = self.archive_objectives[selected_idx]
        else:
            archive_selected = self.archive
            archive_obj_selected = self.archive_objectives

        if len(population) > n_inject:
            fronts = self.fast_non_dominated_sort(objectives)
            if len(fronts) > 0 and len(fronts[-1]) >= n_inject:
                worst_front = fronts[-1]
                crowding = self.crowding_distance(objectives, worst_front)
                worst_idx = np.argsort(crowding)[:n_inject]
                worst_indices = [worst_front[i] for i in worst_idx]
            else:
                worst_indices = np.random.choice(len(population), n_inject, replace=False)
        else:
            worst_indices = np.arange(len(population))

        new_population = population.copy()
        new_objectives = objectives.copy()

        for i, idx in enumerate(worst_indices):
            if i < len(archive_selected):
                new_population[idx] = archive_selected[i]
                new_objectives[idx] = archive_obj_selected[i]

        return new_population, new_objectives

    def calculate_population_diversity(self, objectives: np.ndarray) -> float:
        if len(objectives) <= 1:
            return 0.0

        distances = []
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                distances.append(np.sqrt(np.sum((objectives[i] - objectives[j]) ** 2)))

        return np.mean(distances) if distances else 0.0

    def diversity_preservation_operator(self, population: np.ndarray, objectives: np.ndarray,
                                       generation: int) -> np.ndarray:
        n_var = population.shape[1]

        if len(objectives) > 10:
            obj_ranges = np.max(objectives, axis=0) - np.min(objectives, axis=0)
            avg_range = np.mean(obj_ranges)

            if avg_range < 0.5:
                new_population = population.copy()
                n_perturb = max(1, len(population) // 4)
                perturb_idx = np.random.choice(len(population), n_perturb, replace=False)

                for idx in perturb_idx:
                    perturbation = np.random.normal(0, 0.3, n_var)
                    new_population[idx] = np.clip(population[idx] + perturbation, 0.0, 1.0)

                return new_population

        return population

    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()

        if self.use_enhancements:
            population = self.intelligent_initialization(problem, self.pop_size)
        else:
            population = problem.generate_population(self.pop_size)

        objectives = np.array([problem.evaluate(ind) for ind in population])

        self.archive = population.copy()
        self.archive_objectives = objectives.copy()

        history = {'generation': [], 'best_hv': [], 'best_igd': [], 'avg_objectives': [], 'diversity': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)

        best_hv = 0.0
        best_igd = float('inf')
        no_improvement_count = 0

        for gen in range(self.max_gen):
            if self.use_enhancements:
                exploration_factor = max(0.1, 1.0 - gen / self.max_gen)
                self.adaptive_crossover_rate = self.crossover_prob * (0.5 + 0.5 * exploration_factor)
                self.adaptive_mutation_rate = self.mutation_prob * (1.0 + exploration_factor)

                original_crossover = self.crossover_prob
                original_mutation = self.mutation_prob
                self.crossover_prob = self.adaptive_crossover_rate
                self.mutation_prob = self.adaptive_mutation_rate

            offspring_pop = []

            for _ in range(self.pop_size // 2):
                parent1_idx, parent2_idx = self.adaptive_tournament_selection(population, objectives)
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]

                if np.random.random() < 0.7:
                    child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                else:
                    child1 = self.differential_evolution_crossover(parent1, parent2, population)
                    child2 = self.differential_evolution_crossover(parent2, parent1, population)

                child1 = self.gaussian_mutation(child1, gen, self.max_gen)
                child2 = self.gaussian_mutation(child2, gen, self.max_gen)

                offspring_pop.extend([child1, child2])

            if self.use_enhancements:
                self.crossover_prob = original_crossover
                self.mutation_prob = original_mutation

            offspring_obj = np.array([problem.evaluate(ind) for ind in offspring_pop])

            combined_pop = np.vstack([population, offspring_pop])
            combined_obj = np.vstack([objectives, offspring_obj])

            self.update_archive(combined_pop, combined_obj)

            fronts = self.fast_non_dominated_sort(combined_obj)

            new_pop = []
            new_obj = []
            front_idx = 0

            if len(fronts) > 0:
                first_front = fronts[0]
                if len(first_front) <= self.pop_size:
                    for idx in first_front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                    front_idx += 1

            while len(new_pop) < self.pop_size and front_idx < len(fronts):
                current_front = fronts[front_idx]

                if len(new_pop) + len(current_front) <= self.pop_size:
                    for idx in current_front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    remaining = self.pop_size - len(new_pop)

                    crowding = self.crowding_distance(combined_obj, current_front)

                    novelty = np.zeros(len(current_front))
                    if len(new_obj) > 0:
                        for i, idx in enumerate(current_front):
                            distances = []
                            for obj in new_obj:
                                distances.append(np.sqrt(np.sum((combined_obj[idx] - obj) ** 2)))
                            novelty[i] = np.mean(distances)

                    combined_score = 0.7 * crowding + 0.3 * novelty
                    selected_idx = np.argsort(combined_score)[-remaining:]
                    for idx in selected_idx:
                        new_pop.append(combined_pop[current_front[idx]])
                        new_obj.append(combined_obj[current_front[idx]])

                front_idx += 1

            new_pop_array = np.array(new_pop[:self.pop_size])
            new_obj_array = np.array(new_obj[:self.pop_size])

            if self.use_enhancements and gen % 5 == 0:
                new_pop_array, new_obj_array = self.inject_archive_solutions(
                    new_pop_array, new_obj_array, injection_rate=0.15
                )

            if self.use_enhancements and gen % 10 == 0:
                diversity = self.calculate_population_diversity(new_obj_array)
                if diversity < self.diversity_threshold:
                    new_pop_array = self.diversity_preservation_operator(
                        new_pop_array, new_obj_array, gen
                    )
                    new_obj_array = np.array([problem.evaluate(ind) for ind in new_pop_array])

            population = new_pop_array
            objectives = new_obj_array

            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)

            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
            history['avg_objectives'].append(np.mean(objectives, axis=0))
            history['diversity'].append(self.calculate_population_diversity(objectives))

            if hv > best_hv:
                best_hv = hv
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if self.use_enhancements and no_improvement_count >= self.restart_threshold and gen < self.max_gen * 0.8:
                n_replace = int(self.pop_size * 0.3)
                replace_idx = np.random.choice(self.pop_size, n_replace, replace=False)

                for idx in replace_idx:
                    if len(self.archive) > 0 and np.random.random() < 0.7:
                        archive_idx = np.random.randint(len(self.archive))
                        base = self.archive[archive_idx].copy()
                        mutation = np.random.normal(0, 0.2, problem.n_var)
                        population[idx] = np.clip(base + mutation, 0.0, 1.0)
                    else:
                        population[idx] = problem.generate_population(1)[0]

                objectives = np.array([problem.evaluate(ind) for ind in population])
                no_improvement_count = 0
                self.restart_counter += 1

            if gen > 30 and len(history['best_hv']) > 20:
                recent_hvs = history['best_hv'][-20:]
                improvement = np.max(recent_hvs) - np.min(recent_hvs)
                if improvement < 0.001:
                    break

        runtime = time.time() - start_time

        final_hv = metrics_calc.hypervolume(objectives, ref_point)
        final_igd = metrics_calc.inverted_generational_distance(objectives, true_front)

        return {
            'population': population,
            'objectives': objectives,
            'runtime': runtime,
            'history': history,
            'final_hv': final_hv,
            'final_igd': final_igd,
            'algorithm': self.name,
            'restart_count': self.restart_counter
        }


class SPEA2(BaseMOEA):
    def __init__(self, pop_size: int = 50, max_gen: int = 100):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "SPEA2"

    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()

        population = problem.generate_population(self.pop_size)
        archive = []

        history = {'generation': [], 'best_hv': [], 'best_igd': [], 'avg_objectives': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)

        for gen in range(self.max_gen):
            objectives = np.array([problem.evaluate(ind) for ind in population])

            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)

            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
            history['avg_objectives'].append(np.mean(objectives, axis=0))

            n = len(population)
            strength = np.zeros(n)

            for i in range(n):
                for j in range(n):
                    if i != j and self.dominates(objectives[i], objectives[j]):
                        strength[i] += 1

            raw_fitness = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and self.dominates(objectives[j], objectives[i]):
                        raw_fitness[i] += strength[j]

            archive_indices = []
            for i in range(n):
                is_nondominated = True
                for j in range(n):
                    if i != j and self.dominates(objectives[j], objectives[i]):
                        is_nondominated = False
                        break
                if is_nondominated:
                    archive_indices.append(i)

            archive = [population[i] for i in archive_indices]

            if len(archive) > 0:
                new_pop = []
                for _ in range(self.pop_size):
                    idx = np.random.randint(0, len(archive))
                    individual = archive[idx].copy()
                    individual = self.polynomial_mutation(individual)
                    new_pop.append(individual)

                population = np.array(new_pop)

        objectives = np.array([problem.evaluate(ind) for ind in population])
        runtime = time.time() - start_time

        final_hv = metrics_calc.hypervolume(objectives, ref_point)
        final_igd = metrics_calc.inverted_generational_distance(objectives, true_front)

        return {
            'population': population,
            'objectives': objectives,
            'runtime': runtime,
            'history': history,
            'final_hv': final_hv,
            'final_igd': final_igd,
            'algorithm': self.name
        }


class MOEAD(BaseMOEA):
    def __init__(self, pop_size: int = 50, max_gen: int = 100, neighbor_size: int = 10):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "MOEA/D"
        self.neighbor_size = neighbor_size
        self.weights = None
        self.neighbors = None

    def initialize_weights(self, n_obj: int):
        if n_obj == 2:
            weights = []
            for i in range(self.pop_size):
                w1 = i / (self.pop_size - 1) if self.pop_size > 1 else 0.5
                w2 = 1 - w1
                weights.append([w1, w2])
        else:
            weights = [np.random.dirichlet([1, 1, 1]) for _ in range(self.pop_size)]

        self.weights = np.array(weights)

    def initialize_neighbors(self):
        n = self.pop_size
        self.neighbors = []

        for i in range(n):
            start = max(0, i - self.neighbor_size // 2)
            end = min(n, i + self.neighbor_size // 2 + 1)
            indices = list(range(start, end))
            if i in indices:
                indices.remove(i)
            self.neighbors.append(indices[:self.neighbor_size])

    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()

        self.initialize_weights(problem.n_obj)
        self.initialize_neighbors()

        population = problem.generate_population(self.pop_size)
        objectives = np.array([problem.evaluate(ind) for ind in population])

        ideal_point = np.min(objectives, axis=0)

        history = {'generation': [], 'best_hv': [], 'best_igd': [], 'avg_objectives': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)

        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                neighbor_idx = self.neighbors[i]
                if len(neighbor_idx) >= 2:
                    parents = np.random.choice(neighbor_idx, 2, replace=False)
                    parent1_idx, parent2_idx = parents[0], parents[1]

                    child, _ = self.simulated_binary_crossover(
                        population[parent1_idx], population[parent2_idx]
                    )
                    child = self.polynomial_mutation(child)

                    child_obj = problem.evaluate(child)

                    ideal_point = np.minimum(ideal_point, child_obj)

                    for j in neighbor_idx:
                        current_te = np.max(self.weights[j] * np.abs(objectives[j] - ideal_point))
                        child_te = np.max(self.weights[j] * np.abs(child_obj - ideal_point))

                        if child_te < current_te:
                            population[j] = child
                            objectives[j] = child_obj

            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)

            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
            history['avg_objectives'].append(np.mean(objectives, axis=0))

        runtime = time.time() - start_time

        final_hv = metrics_calc.hypervolume(objectives, ref_point)
        final_igd = metrics_calc.inverted_generational_distance(objectives, true_front)

        return {
            'population': population,
            'objectives': objectives,
            'runtime': runtime,
            'history': history,
            'final_hv': final_hv,
            'final_igd': final_igd,
            'algorithm': self.name
        }


class PerformanceMetrics:
    @staticmethod
    def hypervolume(pareto_front: np.ndarray, ref_point: np.ndarray) -> float:
        if len(pareto_front) == 0:
            return 0.0

        dominated_by_ref = np.all(pareto_front <= ref_point, axis=1)
        if not np.any(dominated_by_ref):
            return 0.0

        front = pareto_front[dominated_by_ref]

        if front.shape[1] == 2:
            sorted_front = front[np.argsort(front[:, 0])]

            pareto_mask = np.ones(len(sorted_front), dtype=bool)
            for i in range(len(sorted_front)):
                for j in range(i + 1, len(sorted_front)):
                    if np.all(sorted_front[j] <= sorted_front[i]):
                        pareto_mask[i] = False
                        break

            pareto_front = sorted_front[pareto_mask]
            if len(pareto_front) == 0:
                return 0.0

            area = 0.0
            prev_x = ref_point[0]

            for point in pareto_front:
                width = prev_x - point[0]
                height = ref_point[1] - point[1]
                area += width * height
                prev_x = point[0]

            return max(area, 0.0)
        else:
            n_samples = 1000
            samples = np.random.rand(n_samples, front.shape[1]) * ref_point

            dominated = 0
            for sample in samples:
                for point in front:
                    if np.all(point <= sample):
                        dominated += 1
                        break

            ref_volume = np.prod(ref_point)
            return (dominated / n_samples) * ref_volume

    @staticmethod
    def inverted_generational_distance(pareto_front: np.ndarray, true_front: np.ndarray) -> float:
        if len(pareto_front) == 0 or len(true_front) == 0:
            return float('inf')

        pareto_front = np.clip(pareto_front, 0, 1000)
        true_front = np.clip(true_front, 0, 1000)

        distances = []
        for true_point in true_front:
            min_dist = np.min(np.sqrt(np.sum((pareto_front - true_point) ** 2, axis=1)))
            distances.append(min_dist)

        return np.mean(distances) if distances else float('inf')

    @staticmethod
    def spread(pareto_front: np.ndarray) -> float:
        if len(pareto_front) < 3:
            return 1.0

        min_vals = np.min(pareto_front, axis=0)
        max_vals = np.max(pareto_front, axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals < 1e-10] = 1.0

        normalized_front = (pareto_front - min_vals) / range_vals

        sorted_front = normalized_front[np.argsort(normalized_front[:, 0])]
        distances = np.sqrt(np.sum((sorted_front[1:] - sorted_front[:-1]) ** 2, axis=1))

        if np.sum(distances) < 1e-10:
            return 1.0

        d_mean = np.mean(distances)
        d_f = np.sqrt(np.sum((sorted_front[0] - sorted_front[-1]) ** 2))

        numerator = d_f + np.sum(np.abs(distances - d_mean))
        denominator = d_f + (len(sorted_front) - 1) * d_mean

        return numerator / denominator if denominator > 0 else 1.0


class BenchmarkFramework:
    def __init__(self):
        self.problems = []
        self.algorithms = {}
        self.results = {}
        self.n_runs = 5
        self.alpha = 0.05

    def load_benchmark_suite(self):
        print("Loading benchmark suite...")

        self.problems.extend([
            ("ZDT1", ZDT1(n_var=10)),
            ("ZDT2", ZDT2(n_var=10)),
            ("ZDT3", ZDT3(n_var=10)),
            ("ZDT4", ZDT4(n_var=5)),
            ("ZDT6", ZDT6(n_var=5)),
            ("DTLZ1", DTLZ1(n_var=5, n_obj=3)),
            ("DTLZ2", DTLZ2(n_var=6, n_obj=3)),
            ("DTLZ3", DTLZ3(n_var=6, n_obj=3)),
            ("DTLZ4", DTLZ4(n_var=6, n_obj=3)),
            ("WFG1", WFG1(n_var=6, n_obj=3)),
            ("WFG2", WFG2(n_var=6, n_obj=3)),
        ])

        print(f"Loaded {len(self.problems)} benchmark problems")

    def setup_algorithms(self):
        print("\nSetting up algorithms...")

        params = {
            'pop_size': 50,
            'max_gen': 100,
            'crossover_prob': 0.9,
            'mutation_prob': 0.2,
            'eta_c': 15.0,
            'eta_m': 20.0
        }

        self.algorithms = {
            'En-NSGA-II': EnhancedNSGA2(pop_size=params['pop_size'], max_gen=params['max_gen'], use_enhancements=True),
            'NSGA-II': NSGA2(pop_size=params['pop_size'], max_gen=params['max_gen']),
            'SPEA2': SPEA2(pop_size=params['pop_size'], max_gen=params['max_gen']),
            'MOEA/D': MOEAD(pop_size=params['pop_size'], max_gen=params['max_gen'], neighbor_size=10)
        }

        print(f"  Population size: {params['pop_size']}")
        print(f"  Max generations: {params['max_gen']}")
        print(f"  Number of runs: {self.n_runs}")

    def run_benchmarks(self):
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE BENCHMARK EXPERIMENT")
        print("="*80)

        self.results = {}
        metrics_calc = PerformanceMetrics()

        total_tests = len(self.problems) * len(self.algorithms) * self.n_runs
        test_count = 0

        for prob_name, problem in self.problems:
            print(f"\n{'-'*60}")
            print(f"Problem: {prob_name} (n_var={problem.n_var}, n_obj={problem.n_obj})")
            print(f"{'-'*60}")

            self.results[prob_name] = {}
            true_front = problem.pareto_front(n_points=100)
            ref_point = problem.get_reference_point()

            for algo_name, algorithm in self.algorithms.items():
                print(f"  Algorithm: {algo_name}")

                algo_results = {
                    'HV': [], 'IGD': [], 'Spread': [], 'Runtime': [],
                    'Convergence_HV': [], 'Convergence_IGD': [],
                    'Final_Populations': [], 'Final_Objectives': []
                }

                for run in range(self.n_runs):
                    test_count += 1
                    progress = test_count / total_tests * 100
                    print(f"    Run {run + 1}/{self.n_runs} [{progress:.1f}%]...", end=' ')

                    np.random.seed(42 * test_count + run)

                    try:
                        result = algorithm.optimize(problem)
                        runtime = result['runtime']

                        pf = result['objectives']
                        pf_positive = np.maximum(pf, 0)

                        hv = result['final_hv']
                        igd = result['final_igd']
                        spread = metrics_calc.spread(pf)

                        algo_results['HV'].append(max(hv, 0))
                        algo_results['IGD'].append(igd)
                        algo_results['Spread'].append(spread)
                        algo_results['Runtime'].append(runtime)

                        if 'history' in result:
                            algo_results['Convergence_HV'].append(result['history']['best_hv'])
                            algo_results['Convergence_IGD'].append(result['history']['best_igd'])

                        algo_results['Final_Populations'].append(result['population'])
                        algo_results['Final_Objectives'].append(result['objectives'])

                        print(f"HV={hv:.4f}, IGD={igd:.4f}, Time={runtime:.1f}s")

                    except Exception as e:
                        print(f"ERROR: {e}")
                        algo_results['HV'].append(0.0)
                        algo_results['IGD'].append(float('inf'))
                        algo_results['Spread'].append(1.0)
                        algo_results['Runtime'].append(float('inf'))
                        algo_results['Convergence_HV'].append([])
                        algo_results['Convergence_IGD'].append([])
                        algo_results['Final_Populations'].append(None)
                        algo_results['Final_Objectives'].append(None)

                self.results[prob_name][algo_name] = {
                    'HV_mean': np.mean(algo_results['HV']),
                    'HV_std': np.std(algo_results['HV']),
                    'IGD_mean': np.mean(algo_results['IGD']),
                    'IGD_std': np.std(algo_results['IGD']),
                    'Spread_mean': np.mean(algo_results['Spread']),
                    'Spread_std': np.std(algo_results['Spread']),
                    'Runtime_mean': np.mean(algo_results['Runtime']),
                    'Runtime_std': np.std(algo_results['Runtime']),
                    'all_HV': algo_results['HV'],
                    'all_IGD': algo_results['IGD'],
                    'all_Spread': algo_results['Spread'],
                    'all_Runtime': algo_results['Runtime'],
                    'convergence_HV': algo_results['Convergence_HV'],
                    'convergence_IGD': algo_results['Convergence_IGD'],
                    'final_populations': algo_results['Final_Populations'],
                    'final_objectives': algo_results['Final_Objectives']
                }

        print(f"\n{'='*80}")
        print("BENCHMARK EXPERIMENT COMPLETED")

        return self.results

    def print_comprehensive_results(self):
        print("\n" + "="*100)
        print("EXPERIMENTAL RESULTS")
        print("="*100)

        problems = sorted(self.results.keys())
        algorithms = list(self.algorithms.keys())

        print("\n" + "="*100)
        print("HYPERVOLUME (HV) RESULTS - Mean ± Standard Deviation")
        print("(Higher values indicate better performance)")
        print("-" * 120)

        header = f"{'Problem':<12} "
        for algo in algorithms:
            header += f"{algo:<25}"
        print(header)
        print("-" * 120)

        for prob_name in problems:
            row = f"{prob_name:<12} "
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    mean = self.results[prob_name][algo_name]['HV_mean']
                    std = self.results[prob_name][algo_name]['HV_std']

                    all_means = [self.results[prob_name][a]['HV_mean'] for a in algorithms if a in self.results[prob_name]]
                    is_best = mean == max(all_means) if all_means else False

                    if is_best:
                        row += f"*{mean:.6f} ± {std:.6f}*{' ':<15}"
                    else:
                        row += f"{mean:.6f} ± {std:.6f}{' ':<15}"
                else:
                    row += f"{'N/A':<25}"
            print(row)

        print("-" * 120)

        print("\nSummary Statistics for HV:")
        print("-" * 80)
        wins = {algo: 0 for algo in algorithms}
        for prob_name in problems:
            best_algo = None
            best_hv = -1
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hv = self.results[prob_name][algo_name]['HV_mean']
                    if hv > best_hv:
                        best_hv = hv
                        best_algo = algo_name
            if best_algo:
                wins[best_algo] += 1

        print(f"Total problems: {len(problems)}")
        for algo in algorithms:
            print(f"{algo}: {wins[algo]} wins ({wins[algo]/len(problems)*100:.1f}%)")

        print("\n\n" + "="*100)
        print("INVERTED GENERATIONAL DISTANCE (IGD) RESULTS - Mean ± Standard Deviation")
        print("(Lower values indicate better performance)")
        print("-" * 120)

        print(header)
        print("-" * 120)

        for prob_name in problems:
            row = f"{prob_name:<12} "
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    mean = self.results[prob_name][algo_name]['IGD_mean']
                    std = self.results[prob_name][algo_name]['IGD_std']

                    all_means = [self.results[prob_name][a]['IGD_mean'] for a in algorithms if a in self.results[prob_name]]
                    all_means = [m for m in all_means if not np.isinf(m)]
                    is_best = mean == min(all_means) if all_means else False

                    if np.isinf(mean):
                        display = "inf"
                    else:
                        display = f"{mean:.6f} ± {std:.6f}"

                    if is_best:
                        row += f"*{display:<25}*"
                    else:
                        row += f"{display:<25}"
                else:
                    row += f"{'N/A':<25}"
            print(row)

        print("-" * 120)

        print("\nSummary Statistics for IGD:")
        print("-" * 80)
        wins = {algo: 0 for algo in algorithms}
        for prob_name in problems:
            best_algo = None
            best_igd = float('inf')
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    igd = self.results[prob_name][algo_name]['IGD_mean']
                    if igd < best_igd and not np.isinf(igd):
                        best_igd = igd
                        best_algo = algo_name
            if best_algo:
                wins[best_algo] += 1

        print(f"Total problems: {len(problems)}")
        for algo in algorithms:
            print(f"{algo}: {wins[algo]} wins ({wins[algo]/len(problems)*100:.1f}%)")

        print("\n\n" + "="*100)
        print("SPREAD METRIC (Δ) RESULTS - Mean ± Standard Deviation")
        print("(Lower values indicate better diversity)")
        print("-" * 120)

        print(header)
        print("-" * 120)

        for prob_name in problems:
            row = f"{prob_name:<12} "
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    mean = self.results[prob_name][algo_name]['Spread_mean']
                    std = self.results[prob_name][algo_name]['Spread_std']

                    all_means = [self.results[prob_name][a]['Spread_mean'] for a in algorithms if a in self.results[prob_name]]
                    is_best = abs(mean - 0) == min([abs(m - 0) for m in all_means]) if all_means else False

                    if is_best:
                        row += f"*{mean:.6f} ± {std:.6f}*{' ':<15}"
                    else:
                        row += f"{mean:.6f} ± {std:.6f}{' ':<15}"
                else:
                    row += f"{'N/A':<25}"
            print(row)

        print("-" * 120)

        print("\n\n" + "="*100)
        print("RUNTIME RESULTS (seconds) - Mean ± Standard Deviation")
        print("(Lower values indicate better efficiency)")
        print("-" * 120)

        print(header)
        print("-" * 120)

        for prob_name in problems:
            row = f"{prob_name:<12} "
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    mean = self.results[prob_name][algo_name]['Runtime_mean']
                    std = self.results[prob_name][algo_name]['Runtime_std']

                    all_means = [self.results[prob_name][a]['Runtime_mean'] for a in algorithms if a in self.results[prob_name]]
                    is_best = mean == min(all_means) if all_means else False

                    if is_best:
                        row += f"*{mean:.3f} ± {std:.3f}*{' ':<15}"
                    else:
                        row += f"{mean:.3f} ± {std:.3f}{' ':<15}"
                else:
                    row += f"{'N/A':<25}"
            print(row)

        print("-" * 120)

        print("\n\n" + "="*100)
        print("COMPREHENSIVE PERFORMANCE SUMMARY")
        print("="*100)
        print("\nTable: Algorithm Performance Ranking Across All Metrics")
        print("-" * 120)
        print(f"{'Algorithm':<15} {'HV Wins':<10} {'IGD Wins':<10} {'Spread Wins':<12} {'Runtime Wins':<12} {'Total Wins':<10} {'Win Rate (%)':<12}")
        print("-" * 120)

        for algo in algorithms:
            hv_wins = 0
            igd_wins = 0
            spread_wins = 0
            runtime_wins = 0

            for prob_name in problems:
                best_hv = -1
                best_hv_algo = None
                for a in algorithms:
                    if a in self.results[prob_name]:
                        hv = self.results[prob_name][a]['HV_mean']
                        if hv > best_hv:
                            best_hv = hv
                            best_hv_algo = a
                if best_hv_algo == algo:
                    hv_wins += 1

                best_igd = float('inf')
                best_igd_algo = None
                for a in algorithms:
                    if a in self.results[prob_name]:
                        igd = self.results[prob_name][a]['IGD_mean']
                        if igd < best_igd and not np.isinf(igd):
                            best_igd = igd
                            best_igd_algo = a
                if best_igd_algo == algo:
                    igd_wins += 1

                best_spread = float('inf')
                best_spread_algo = None
                for a in algorithms:
                    if a in self.results[prob_name]:
                        spread = self.results[prob_name][a]['Spread_mean']
                        if abs(spread - 0) < best_spread:
                            best_spread = abs(spread - 0)
                            best_spread_algo = a
                if best_spread_algo == algo:
                    spread_wins += 1

                best_runtime = float('inf')
                best_runtime_algo = None
                for a in algorithms:
                    if a in self.results[prob_name]:
                        runtime = self.results[prob_name][a]['Runtime_mean']
                        if runtime < best_runtime:
                            best_runtime = runtime
                            best_runtime_algo = a
                if best_runtime_algo == algo:
                    runtime_wins += 1

            total_wins = hv_wins + igd_wins + spread_wins + runtime_wins
            total_possible = 4 * len(problems)
            win_rate = (total_wins / total_possible) * 100 if total_possible > 0 else 0

            print(f"{algo:<15} {hv_wins:<10} {igd_wins:<10} {spread_wins:<12} {runtime_wins:<12} {total_wins:<10} {win_rate:<12.1f}")

        print("-" * 120)

    def create_pareto_front_comparisons(self):
        print("\n" + "="*100)
        print("COMPARISON OF THE PARETO FRONTS")
        print("="*100)

        os.makedirs('analysis_plots/pareto_fronts', exist_ok=True)

        zdt_problems = [p for p in self.problems if p[1].name.startswith('ZDT')]
        dtlz_problems = [p for p in self.problems if p[1].name.startswith('DTLZ')]
        wfg_problems = [p for p in self.problems if p[1].name.startswith('WFG')]

        algorithms = list(self.algorithms.keys())
        colors = {'En-NSGA-II': 'orange', 'NSGA-II': 'blue', 'SPEA2': 'green', 'MOEA/D': 'red'}
        markers = {'En-NSGA-II': 'o', 'NSGA-II': 's', 'SPEA2': '^', 'MOEA/D': 'D'}

        print("\nZDT Test Suite")
        print("-" * 80)

        for prob_name, problem in zdt_problems:
            print(f"  Creating plot for {prob_name}...")

            plt.figure(figsize=(12, 8))

            true_front = problem.pareto_front(100)
            plt.scatter(true_front[:, 0], true_front[:, 1], color='black', s=30, alpha=0.7,
                       label='True Pareto Front', zorder=5, marker='x')

            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hvs = self.results[prob_name][algo_name]['all_HV']
                    if hvs:
                        best_run_idx = np.argmax(hvs)
                        objectives = self.results[prob_name][algo_name]['final_objectives'][best_run_idx]

                        if objectives is not None and len(objectives) > 0:
                            plt.scatter(objectives[:, 0], objectives[:, 1], color=colors[algo_name],
                                       s=40, alpha=0.6, label=algo_name, marker=markers[algo_name], zorder=4)

            plt.xlabel('$f_1$', fontsize=14, fontweight='bold')
            plt.ylabel('$f_2$', fontsize=14, fontweight='bold')
            plt.title(f'Pareto Front Comparison - {prob_name}', fontsize=16, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3, linestyle='--')

            plt.tight_layout()
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.pdf', bbox_inches='tight')
            plt.close()

        print("\nDTLZ Test Suite")
        print("-" * 80)

        for prob_name, problem in dtlz_problems:
            print(f"  Creating plot for {prob_name}...")

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            true_front = problem.pareto_front(200)
            ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2],
                      color='black', s=20, alpha=0.4, label='True Pareto Front', marker='o')

            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hvs = self.results[prob_name][algo_name]['all_HV']
                    if hvs:
                        best_run_idx = np.argmax(hvs)
                        objectives = self.results[prob_name][algo_name]['final_objectives'][best_run_idx]

                        if objectives is not None and len(objectives) > 0:
                            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                                      color=colors[algo_name], s=40, alpha=0.7, label=algo_name, marker=markers[algo_name])

            ax.set_xlabel('$f_1$', fontsize=12, fontweight='bold')
            ax.set_ylabel('$f_2$', fontsize=12, fontweight='bold')
            ax.set_zlabel('$f_3$', fontsize=12, fontweight='bold')
            ax.set_title(f'Pareto Front Comparison - {prob_name}', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=11)

            plt.tight_layout()
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.pdf', bbox_inches='tight')
            plt.close()

        print("\nWFG Test Suite")
        print("-" * 80)

        for prob_name, problem in wfg_problems:
            print(f"  Creating plot for {prob_name}...")

            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            true_front = problem.pareto_front(200)
            ax.scatter(true_front[:, 0], true_front[:, 1], true_front[:, 2],
                      color='black', s=20, alpha=0.4, label='True Pareto Front', marker='o')

            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hvs = self.results[prob_name][algo_name]['all_HV']
                    if hvs:
                        best_run_idx = np.argmax(hvs)
                        objectives = self.results[prob_name][algo_name]['final_objectives'][best_run_idx]

                        if objectives is not None and len(objectives) > 0:
                            ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2],
                                      color=colors[algo_name], s=40, alpha=0.7, label=algo_name, marker=markers[algo_name])

            ax.set_xlabel('$f_1$', fontsize=12, fontweight='bold')
            ax.set_ylabel('$f_2$', fontsize=12, fontweight='bold')
            ax.set_zlabel('$f_3$', fontsize=12, fontweight='bold')
            ax.set_title(f'Pareto Front Comparison - {prob_name}', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=11)

            plt.tight_layout()
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'analysis_plots/pareto_fronts/{prob_name}_pareto_front.pdf', bbox_inches='tight')
            plt.close()

        print("\n✓ All Pareto front comparison plots saved to 'analysis_plots/pareto_fronts/'")

    def statistical_analysis(self):
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)

        problems = sorted(self.results.keys())

        print("\n1. Wilcoxon Rank-Sum Test (En-NSGA-II vs Baselines) - HV Metric")
        print("-" * 100)
        print(f"{'Problem':<10} {'vs NSGA-II':<15} {'vs SPEA2':<15} {'vs MOEA/D':<15}")
        print(f"{'':<10} {'p-value':<15} {'p-value':<15} {'p-value':<15}")
        print("-" * 100)

        for prob_name in problems:
            if 'En-NSGA-II' not in self.results[prob_name]:
                continue

            en_hvs = self.results[prob_name]['En-NSGA-II']['all_HV']

            results = []
            for baseline in ['NSGA-II', 'SPEA2', 'MOEA/D']:
                if baseline in self.results[prob_name]:
                    base_hvs = self.results[prob_name][baseline]['all_HV']
                    if len(en_hvs) >= 3 and len(base_hvs) >= 3:
                        try:
                            _, p_value = stats.ranksums(en_hvs, base_hvs)
                            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            results.append(f"{p_value:.6f}{sig}")
                        except:
                            results.append("N/A")
                    else:
                        results.append("N/A")
                else:
                    results.append("N/A")

            print(f"{prob_name:<10} {results[0]:<15} {results[1]:<15} {results[2]:<15}")

        print("\n2. Wilcoxon Rank-Sum Test (En-NSGA-II vs Baselines) - IGD Metric")
        print("-" * 100)
        print(f"{'Problem':<10} {'vs NSGA-II':<15} {'vs SPEA2':<15} {'vs MOEA/D':<15}")
        print(f"{'':<10} {'p-value':<15} {'p-value':<15} {'p-value':<15}")
        print("-" * 100)

        for prob_name in problems:
            if 'En-NSGA-II' not in self.results[prob_name]:
                continue

            en_igds = self.results[prob_name]['En-NSGA-II']['all_IGD']

            results = []
            for baseline in ['NSGA-II', 'SPEA2', 'MOEA/D']:
                if baseline in self.results[prob_name]:
                    base_igds = self.results[prob_name][baseline]['all_IGD']
                    if len(en_igds) >= 3 and len(base_igds) >= 3:
                        try:
                            _, p_value = stats.ranksums(en_igds, base_igds)
                            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            results.append(f"{p_value:.6f}{sig}")
                        except:
                            results.append("N/A")
                    else:
                        results.append("N/A")
                else:
                    results.append("N/A")

            print(f"{prob_name:<10} {results[0]:<15} {results[1]:<15} {results[2]:<15}")

        print("\nNote: *** p < 0.001, ** p < 0.01, * p < 0.05")

    def save_results_to_csv(self):
        print("\n" + "="*80)
        print("SAVING RESULTS TO CSV FILES")
        print("="*80)

        detailed_data = []
        for prob_name, algo_results in self.results.items():
            for algo_name, metrics in algo_results.items():
                for run in range(self.n_runs):
                    if run < len(metrics['all_HV']):
                        detailed_data.append({
                            'Problem': prob_name,
                            'Algorithm': algo_name,
                            'Run': run + 1,
                            'HV': metrics['all_HV'][run],
                            'IGD': metrics['all_IGD'][run] if run < len(metrics['all_IGD']) else np.nan,
                            'Spread': metrics['all_Spread'][run] if run < len(metrics['all_Spread']) else np.nan,
                            'Runtime': metrics['all_Runtime'][run] if run < len(metrics['all_Runtime']) else np.nan
                        })

        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_csv('complete_benchmark_results.csv', index=False)
            print(f"Detailed results saved to 'complete_benchmark_results.csv' ({len(df_detailed)} rows)")

        summary_data = []
        for prob_name, algo_results in self.results.items():
            for algo_name, metrics in algo_results.items():
                summary_data.append({
                    'Problem': prob_name,
                    'Algorithm': algo_name,
                    'HV_Mean': metrics['HV_mean'],
                    'HV_Std': metrics['HV_std'],
                    'IGD_Mean': metrics['IGD_mean'],
                    'IGD_Std': metrics['IGD_std'],
                    'Spread_Mean': metrics['Spread_mean'],
                    'Spread_Std': metrics['Spread_std'],
                    'Runtime_Mean': metrics['Runtime_mean'],
                    'Runtime_Std': metrics['Runtime_std']
                })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv('complete_summary_results.csv', index=False)
            print(f"Summary results saved to 'complete_summary_results.csv' ({len(summary_data)} rows)")

        stats_data = []
        for prob_name in sorted(self.results.keys()):
            if 'En-NSGA-II' not in self.results[prob_name]:
                continue

            en_hvs = self.results[prob_name]['En-NSGA-II']['all_HV']
            en_igds = self.results[prob_name]['En-NSGA-II']['all_IGD']

            for baseline in ['NSGA-II', 'SPEA2', 'MOEA/D']:
                if baseline in self.results[prob_name]:
                    base_hvs = self.results[prob_name][baseline]['all_HV']
                    base_igds = self.results[prob_name][baseline]['all_IGD']

                    if len(en_hvs) >= 3 and len(base_hvs) >= 3:
                        try:
                            _, hv_p = stats.ranksums(en_hvs, base_hvs)
                        except:
                            hv_p = np.nan

                        try:
                            _, igd_p = stats.ranksums(en_igds, base_igds)
                        except:
                            igd_p = np.nan

                        stats_data.append({
                            'Problem': prob_name,
                            'Comparison': f'En-NSGA-II vs {baseline}',
                            'HV_p_value': hv_p,
                            'IGD_p_value': igd_p,
                            'HV_significant': hv_p < 0.05 if not np.isnan(hv_p) else False,
                            'IGD_significant': igd_p < 0.05 if not np.isnan(igd_p) else False
                        })

        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            df_stats.to_csv('statistical_significance.csv', index=False)
            print(f"Statistical significance results saved to 'statistical_significance.csv'")

        print("\nAll results have been saved successfully!")


def main():
    print("\n" + "="*100)
    print("ENHANCED NSGA-II COMPREHENSIVE BENCHMARK ANALYSIS")
    print("="*100)

    framework = BenchmarkFramework()
    framework.load_benchmark_suite()
    framework.setup_algorithms()

    try:
        print("\nStarting comprehensive benchmark analysis...")
        framework.run_benchmarks()

        print("\nGenerating comprehensive analysis...")
        framework.print_comprehensive_results()
        framework.statistical_analysis()
        framework.create_pareto_front_comparisons()
        framework.save_results_to_csv()

    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*100)
    print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*100)


if __name__ == "__main__":
    import time as timer
    start_time = timer.time()

    main()

    total_time = timer.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("✓ Results saved to CSV files")
    print("✓ Pareto front comparison plots saved to 'analysis_plots/pareto_fronts/' directory")