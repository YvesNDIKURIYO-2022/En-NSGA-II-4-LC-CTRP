import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    print("Gurobi not available, using simplified exact method")
    GUROBI_AVAILABLE = False


def generate_instance_s1_8nodes():
    coordinates = {
        0: (35, 35), 1: (41, 49), 2: (35, 17), 3: (55, 45),
        4: (55, 20), 5: (15, 30), 6: (25, 30), 7: (20, 50)
    }
    return generate_instance_from_coordinates(coordinates, 0, 7, max_time=24, max_emission=50)


def generate_instance_s2_12nodes():
    coordinates = {
        0: (40, 40), 1: (45, 68), 2: (45, 70), 3: (42, 66), 4: (42, 68),
        5: (40, 69), 6: (38, 68), 7: (38, 70), 8: (35, 66), 9: (35, 69),
        10: (25, 85), 11: (22, 75)
    }
    return generate_instance_from_coordinates(coordinates, 0, 11, max_time=36, max_emission=80)


def generate_instance_s3_15nodes():
    coordinates = {
        0: (82, 76), 1: (96, 44), 2: (50, 5), 3: (49, 8), 4: (13, 7),
        5: (29, 89), 6: (58, 30), 7: (84, 39), 8: (14, 24), 9: (2, 39),
        10: (3, 82), 11: (5, 10), 12: (98, 52), 13: (84, 25), 14: (61, 59)
    }
    return generate_instance_from_coordinates(coordinates, 0, 14, max_time=48, max_emission=120)


def generate_instance_s4_18nodes():
    coordinates = {
        0: (40, 50), 1: (25, 85), 2: (22, 75), 3: (22, 85), 4: (20, 80),
        5: (20, 85), 6: (18, 75), 7: (15, 75), 8: (15, 80), 9: (30, 50),
        10: (30, 52), 11: (28, 52), 12: (28, 55), 13: (25, 50), 14: (25, 52),
        15: (23, 52), 16: (23, 55), 17: (10, 35)
    }
    return generate_instance_from_coordinates(coordinates, 0, 17, max_time=60, max_emission=150)


def generate_instance_from_coordinates(coordinates, origin, destination, max_time, max_emission):
    n_nodes = len(coordinates)
    nodes = list(range(n_nodes))
    
    distance = np.zeros((n_nodes, n_nodes))
    for i in nodes:
        for j in nodes:
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance[i][j] = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    mode_params = {
        'road': {'cost_per_km': 0.23, 'emission_per_km': 0.120, 'speed': 70, 'fixed_cost': 10, 'capacity': 2},
        'rail': {'cost_per_km': 0.09, 'emission_per_km': 0.020, 'speed': 60, 'fixed_cost': 50, 'capacity': 5}
    }
    
    costs = {'road': {}, 'rail': {}}
    emissions = {'road': {}, 'rail': {}}
    times = {'road': {}, 'rail': {}}
    
    for mode in ['road', 'rail']:
        params = mode_params[mode]
        for i in nodes:
            for j in nodes:
                if i != j:
                    dist = distance[i][j]
                    costs[mode][(i, j)] = params['fixed_cost'] + params['cost_per_km'] * dist
                    emissions[mode][(i, j)] = params['emission_per_km'] * dist
                    times[mode][(i, j)] = dist / params['speed']
    
    return {
        'n_nodes': n_nodes, 'nodes': nodes, 'origin': origin, 'destination': destination,
        'demand': 1.0, 'costs': costs, 'emissions': emissions, 'times': times,
        'max_time': max_time, 'max_emission': max_emission,
        'capacity': {'road': mode_params['road']['capacity'], 'rail': mode_params['rail']['capacity']},
        'distance': distance, 'coordinates': coordinates
    }


class EnhancedENNSGAII:
    def __init__(self, instance, pop_size=30, n_gen=100):
        self.instance = instance
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.population = []
        self.best_history = []
        self.diversity_history = []
        
    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            path, modes = self.generate_diverse_solution()
            cost, emission, time, feasible = self.evaluate_solution(path, modes)
            self.population.append({
                'path': path, 'modes': modes, 'cost': cost, 'emission': emission,
                'time': time, 'feasible': feasible,
                'fitness': self.calculate_fitness(cost, emission, time, feasible)
            })
    
    def generate_diverse_solution(self):
        strategy = np.random.choice(['greedy', 'random', 'balanced'])
        if strategy == 'greedy':
            return self.generate_greedy_path()
        elif strategy == 'random':
            return self.generate_random_path()
        else:
            return self.generate_balanced_path()
    
    def generate_greedy_path(self):
        n_nodes = self.instance['n_nodes']
        origin = self.instance['origin']
        destination = self.instance['destination']
        
        path = [origin]
        current = origin
        
        while current != destination and len(path) < n_nodes:
            candidates = [j for j in range(n_nodes) if j != current and j not in path]
            if not candidates:
                break
            best_node = min(candidates, key=lambda j: self.instance['distance'][j][destination])
            path.append(best_node)
            current = best_node
        
        if path[-1] != destination:
            path.append(destination)
        
        modes = ['rail' if self.instance['distance'][path[i]][path[i+1]] > 50 else 'road' 
                for i in range(len(path) - 1)]
        return path, modes
    
    def generate_random_path(self):
        n_nodes = self.instance['n_nodes']
        origin = self.instance['origin']
        destination = self.instance['destination']
        
        path = [origin]
        current = origin
        
        while current != destination and len(path) < n_nodes:
            candidates = [j for j in range(n_nodes) if j != current and j not in path]
            if not candidates:
                if len(path) > 1:
                    path.pop()
                    current = path[-1]
                    continue
                else:
                    break
            next_node = np.random.choice(candidates)
            path.append(next_node)
            current = next_node
        
        if path[-1] != destination:
            path.append(destination)
        
        modes = [np.random.choice(['road', 'rail']) for _ in range(len(path) - 1)]
        return path, modes
    
    def generate_balanced_path(self):
        n_nodes = self.instance['n_nodes']
        origin = self.instance['origin']
        destination = self.instance['destination']
        
        path = [origin]
        current = origin
        
        while current != destination and len(path) < n_nodes:
            candidates = [j for j in range(n_nodes) if j != current and j not in path]
            if not candidates:
                break
            
            scores = []
            for j in candidates:
                progress = self.instance['distance'][current][destination] - self.instance['distance'][j][destination]
                exploration = len([k for k in range(n_nodes) if k not in path + [j]]) / n_nodes
                score = 0.7 * progress + 0.3 * exploration
                scores.append(score)
            
            next_node = candidates[np.argmax(scores)]
            path.append(next_node)
            current = next_node
        
        if path[-1] != destination:
            path.append(destination)
        
        modes = []
        for i in range(len(path) - 1):
            cost_road = self.instance['costs']['road'].get((path[i], path[i+1]), np.inf)
            cost_rail = self.instance['costs']['rail'].get((path[i], path[i+1]), np.inf)
            modes.append('rail' if cost_rail < cost_road * 0.8 else 'road')
        
        return path, modes
    
    def evaluate_solution(self, path, modes):
        total_cost = 0
        total_emission = 0
        total_time = 0
        feasible = True
        
        for idx in range(len(path) - 1):
            i, j = path[idx], path[idx + 1]
            mode = modes[idx]
            
            if (i, j) not in self.instance['costs'][mode]:
                feasible = False
                break
            
            total_cost += self.instance['costs'][mode][(i, j)]
            total_emission += self.instance['emissions'][mode][(i, j)]
            total_time += self.instance['times'][mode][(i, j)]
        
        if total_time > self.instance['max_time'] or total_emission > self.instance['max_emission']:
            feasible = False
        
        return total_cost, total_emission, total_time, feasible
    
    def calculate_fitness(self, cost, emission, time, feasible):
        if not feasible:
            return 1e6
        return cost * 0.7 + emission * 0.2 + time * 0.1
    
    def crossover(self, parent1, parent2):
        path1, modes1 = parent1['path'], parent1['modes']
        path2, modes2 = parent2['path'], parent2['modes']
        
        if len(path1) < 3 or len(path2) < 3:
            return parent1['path'].copy(), parent1['modes'].copy()
        
        cross_point = np.random.randint(1, min(len(path1), len(path2)) - 1)
        
        child_path = path1[:cross_point] + path2[cross_point:]
        child_modes = modes1[:cross_point] + modes2[cross_point:]
        
        seen = set()
        unique_path = []
        unique_modes = []
        for node, mode in zip(child_path, child_modes + [modes2[-1] if len(modes2) > cross_point else 'road']):
            if node not in seen:
                seen.add(node)
                unique_path.append(node)
                unique_modes.append(mode)
        
        if unique_path[0] != self.instance['origin']:
            unique_path.insert(0, self.instance['origin'])
            unique_modes.insert(0, 'road')
        
        if unique_path[-1] != self.instance['destination']:
            unique_path.append(self.instance['destination'])
        
        unique_modes = unique_modes[:len(unique_path)-1]
        if len(unique_modes) < len(unique_path) - 1:
            unique_modes += ['road'] * (len(unique_path) - 1 - len(unique_modes))
        
        return unique_path, unique_modes
    
    def mutate(self, path, modes):
        new_path = path.copy()
        new_modes = modes.copy()
        
        if np.random.random() < 0.3 and len(new_path) > 3:
            idx1, idx2 = np.random.choice(range(1, len(new_path)-1), 2, replace=False)
            new_path[idx1], new_path[idx2] = new_path[idx2], new_path[idx1]
        
        if np.random.random() < 0.4 and len(new_modes) > 0:
            idx = np.random.randint(0, len(new_modes))
            new_modes[idx] = 'rail' if new_modes[idx] == 'road' else 'road'
        
        return new_path, new_modes
    
    def calculate_diversity(self):
        if len(self.population) == 0:
            return 0
        costs = [sol['cost'] for sol in self.population if sol['feasible']]
        if len(costs) < 2:
            return 0
        return np.std(costs) / (np.mean(costs) + 1e-10)
    
    def run(self, seed=42):
        np.random.seed(seed)
        self.initialize_population()
        
        for gen in range(self.n_gen):
            new_population = []
            
            self.population.sort(key=lambda x: x['fitness'])
            elite_count = max(1, int(self.pop_size * 0.2))
            new_population.extend(self.population[:elite_count])
            
            while len(new_population) < self.pop_size:
                idx1, idx2 = np.random.choice(range(len(self.population)), 2, replace=False)
                parent1 = self.population[idx1]
                parent2 = self.population[idx2]
                
                child_path, child_modes = self.crossover(parent1, parent2)
                child_path, child_modes = self.mutate(child_path, child_modes)
                cost, emission, time, feasible = self.evaluate_solution(child_path, child_modes)
                
                new_population.append({
                    'path': child_path, 'modes': child_modes, 'cost': cost,
                    'emission': emission, 'time': time, 'feasible': feasible,
                    'fitness': self.calculate_fitness(cost, emission, time, feasible)
                })
            
            self.population = new_population
            self.population.sort(key=lambda x: x['fitness'])
            
            feasible_solutions = [sol for sol in self.population if sol['feasible']]
            if feasible_solutions:
                best_solution = min(feasible_solutions, key=lambda x: x['cost'])
                self.best_history.append({
                    'generation': gen, 'best_cost': best_solution['cost'],
                    'best_emission': best_solution['emission'], 'best_time': best_solution['time']
                })
            else:
                self.best_history.append({
                    'generation': gen, 'best_cost': np.inf, 'best_emission': np.inf, 'best_time': np.inf
                })
            
            self.diversity_history.append(self.calculate_diversity())
        
        feasible_solutions = [sol for sol in self.population if sol['feasible']]
        if feasible_solutions:
            best_solution = min(feasible_solutions, key=lambda x: x['cost'])
            best_cost = best_solution['cost']
            
            pareto_front = []
            for sol in feasible_solutions:
                dominated = False
                for other in feasible_solutions:
                    if (other['cost'] <= sol['cost'] and other['emission'] <= sol['emission'] and
                        other['time'] <= sol['time'] and (other['cost'] < sol['cost'] or 
                        other['emission'] < sol['emission'] or other['time'] < sol['time'])):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append([sol['cost'], sol['emission'], sol['time']])
            
            return {'best_cost': best_cost, 'pareto_front': np.array(pareto_front),
                    'convergence': self.best_history, 'diversity': self.diversity_history}
        else:
            return {'best_cost': np.inf, 'pareto_front': np.array([]),
                    'convergence': [], 'diversity': []}


def plot_individual_convergence(instance_name, convergence_data, exact_cost, save=True):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    for run_idx, run_data in enumerate(convergence_data):
        if len(run_data) > 0:
            gens = [d['generation'] for d in run_data]
            costs = [d['best_cost'] for d in run_data]
            ax1.plot(gens, costs, alpha=0.3, linewidth=0.8, 
                    color=plt.cm.viridis(run_idx/len(convergence_data)))
    
    ax1.axhline(y=exact_cost, color='red', linestyle='--', linewidth=2, label=f'Exact: ${exact_cost:.2f}')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Cost ($)')
    ax1.set_title(f'{instance_name} - Cost Convergence (Individual Runs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    max_gen = max([max([d['generation'] for d in run_data]) for run_data in convergence_data if len(run_data) > 0])
    
    mean_costs, std_costs = [], []
    generations = list(range(0, max_gen + 1, 5))
    
    for gen in generations:
        gen_costs = []
        for run_data in convergence_data:
            if len(run_data) > 0:
                gen_diffs = [abs(d['generation'] - gen) for d in run_data]
                if gen_diffs and min(gen_diffs) <= 5:
                    closest_idx = np.argmin(gen_diffs)
                    gen_costs.append(run_data[closest_idx]['best_cost'])
        
        if gen_costs:
            mean_costs.append(np.mean(gen_costs))
            std_costs.append(np.std(gen_costs))
        else:
            mean_costs.append(np.nan)
            std_costs.append(np.nan)
    
    ax2.plot(generations[:len(mean_costs)], mean_costs, 'b-', linewidth=2, label='Mean Cost')
    ax2.fill_between(generations[:len(mean_costs)], 
                    np.array(mean_costs) - np.array(std_costs),
                    np.array(mean_costs) + np.array(std_costs),
                    alpha=0.2, color='blue', label='±1 STD')
    ax2.axhline(y=exact_cost, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Cost ($)')
    ax2.set_title(f'{instance_name} - Mean Convergence with CI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    success_rates = []
    for gen in generations:
        success_count, total_count = 0, 0
        for run_data in convergence_data:
            if len(run_data) > 0:
                gen_diffs = [abs(d['generation'] - gen) for d in run_data]
                if gen_diffs and min(gen_diffs) <= 5:
                    closest_idx = np.argmin(gen_diffs)
                    total_count += 1
                    if abs(run_data[closest_idx]['best_cost'] - exact_cost) / exact_cost * 100 <= 1.0:
                        success_count += 1
        success_rates.append(success_count / total_count * 100 if total_count > 0 else np.nan)
    
    ax3.plot(generations[:len(success_rates)], success_rates, 'g-', linewidth=2)
    ax3.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title(f'{instance_name} - Success Rate (≤1% gap)')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    final_costs = [run_data[-1]['best_cost'] for run_data in convergence_data if len(run_data) > 0]
    
    if final_costs:
        ax4.hist(final_costs, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        ax4.axvline(x=exact_cost, color='red', linestyle='--', linewidth=2, label=f'Exact: ${exact_cost:.2f}')
        ax4.axvline(x=np.mean(final_costs), color='green', linestyle='-', linewidth=2, label=f'Mean: ${np.mean(final_costs):.2f}')
        
        if len(final_costs) > 1:
            from scipy.stats import norm
            mu, std = np.mean(final_costs), np.std(final_costs)
            xmin, xmax = ax4.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            ax4.plot(x, norm.pdf(x, mu, std), 'k', linewidth=1.5, label='Normal fit')
        
        ax4.set_xlabel('Final Cost ($)')
        ax4.set_ylabel('Density')
        ax4.set_title(f'{instance_name} - Final Cost Distribution (n={len(final_costs)})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{instance_name} - Comprehensive Validation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save:
        filename = f'convergence_{instance_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    plt.show()


def plot_instance_comparison_summary(results_summary):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    instances = results_summary['instance']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    x_pos = np.arange(len(instances))
    
    ax1 = axes[0, 0]
    mean_gaps = results_summary['stat_mean_gap_%']
    std_gaps = results_summary['stat_std_gap_%']
    bars = ax1.bar(x_pos, mean_gaps, yerr=std_gaps, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, mean_gap in zip(bars, mean_gaps):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{mean_gap:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([inst.split()[0] for inst in instances], fontsize=10)
    ax1.set_ylabel('Optimality Gap (%)', fontweight='bold')
    ax1.set_title('Mean Optimality Gap by Instance', fontweight='bold', pad=15)
    ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='1% threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2 = axes[0, 1]
    success_rates = results_summary['stat_success_rate_1%']
    bars = ax2.bar(x_pos, success_rates, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, rate in zip(bars, success_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([inst.split()[0] for inst in instances], fontsize=10)
    ax2.set_ylabel('Success Rate (%)', fontweight='bold')
    ax2.set_title('Success Rate (≤1% gap) by Instance', fontweight='bold', pad=15)
    ax2.set_ylim([0, 105])
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% target')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = axes[1, 0]
    mean_times = results_summary['ennsga_mean_time_s']
    std_times = results_summary['ennsga_std_time_s']
    bars = ax3.bar(x_pos, mean_times, yerr=std_times, capsize=10, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, mean_time in zip(bars, mean_times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                f'{mean_time:.3f}s', ha='center', va='bottom', fontsize=9)
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([inst.split()[0] for inst in instances], fontsize=10)
    ax3.set_ylabel('Computation Time (seconds)', fontweight='bold')
    ax3.set_title('Mean Computation Time by Instance', fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    
    ax4 = axes[1, 1]
    nodes = results_summary['nodes']
    optimality_gaps = results_summary['stat_mean_gap_%']
    
    ax4.plot(nodes, optimality_gaps, 'o-', linewidth=2, markersize=10,
            color='#d62728', markerfacecolor='white', markeredgewidth=2)
    
    z = np.polyfit(nodes, optimality_gaps, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(min(nodes), max(nodes), 100)
    ax4.plot(x_smooth, p(x_smooth), '--', color='gray', alpha=0.7, label='Quadratic trend')
    
    for i, (node, gap) in enumerate(zip(nodes, optimality_gaps)):
        ax4.annotate(f'{instances[i].split()[0]}\n{gap:.1f}%', (node, gap), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    ax4.set_xlabel('Number of Nodes', fontweight='bold')
    ax4.set_ylabel('Optimality Gap (%)', fontweight='bold')
    ax4.set_title('Algorithm Scalability Analysis', fontweight='bold', pad=15)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Instance Validation Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('instance_comparison_summary.png', dpi=300, bbox_inches='tight')
    print("Saved: instance_comparison_summary.png")
    plt.show()


def solve_exact_cost_simplified(instance):
    if not GUROBI_AVAILABLE:
        min_cost = float('inf')
        if (origin, destination) in instance['costs']['road']:
            min_cost = min(min_cost, instance['costs']['road'][(origin, destination)])
        if (origin, destination) in instance['costs']['rail']:
            min_cost = min(min_cost, instance['costs']['rail'][(origin, destination)])
        if min_cost == float('inf'):
            dist = instance['distance'][origin][destination]
            min_cost = 10 + 0.09 * dist
        return {'status': 'estimated', 'cost': min_cost}
    
    n_nodes, origin, destination = instance['n_nodes'], instance['origin'], instance['destination']
    model = gp.Model("Exact-Cost")
    model.setParam('OutputFlag', 0)
    
    x = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                for m_idx in [0, 1]:
                    x[i,j,m_idx] = model.addVar(vtype=GRB.BINARY)
    
    model.addConstr(gp.quicksum(x[origin,j,m] for j in range(n_nodes) if j != origin for m in [0,1]) == 1)
    model.addConstr(gp.quicksum(x[j,destination,m] for j in range(n_nodes) if j != destination for m in [0,1]) == 1)
    
    for i in range(n_nodes):
        if i not in [origin, destination]:
            flow_in = gp.quicksum(x[j,i,m] for j in range(n_nodes) if j != i for m in [0,1])
            flow_out = gp.quicksum(x[i,j,m] for j in range(n_nodes) if j != i for m in [0,1])
            model.addConstr(flow_in == flow_out)
    
    obj = gp.quicksum(instance['costs'][mode][(i,j)] * x[i,j,m_idx]
          for i in range(n_nodes) for j in range(n_nodes) if i != j
          for m_idx, mode in enumerate(['road', 'rail']))
    
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return {'status': 'optimal', 'cost': model.objVal}
    return {'status': 'failed', 'cost': np.nan}


def run_validation_with_individual_plots():
    print("=" * 80)
    print("INDIVIDUAL INSTANCE VALIDATION WITH DETAILED PLOTS")
    print("=" * 80)
    
    instances = {
        'S1 (8 nodes)': generate_instance_s1_8nodes(),
        'S2 (12 nodes)': generate_instance_s2_12nodes(),
        'S3 (15 nodes)': generate_instance_s3_15nodes(),
        'S4 (18 nodes)': generate_instance_s4_18nodes()
    }
    
    results = []
    all_convergence_data = []
    
    for instance_name, instance in instances.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING: {instance_name}")
        print(f"{'='*60}")
        
        print("  Calculating exact minimum cost...")
        exact_result = solve_exact_cost_simplified(instance)
        exact_cost = exact_result['cost']
        print(f"    {'Exact' if exact_result['status'] == 'optimal' else 'Estimated'} cost: ${exact_cost:.2f}")
        
        print(f"  Running Enhanced En-NSGA-II (10 seeds)...")
        start_time = time.time()
        
        best_costs = []
        convergence_data = []
        run_times = []
        
        for seed in range(10):
            print(f"    Run {seed+1}/10", end='\r')
            seed_start = time.time()
            algorithm = EnhancedENNSGAII(instance, pop_size=30, n_gen=100)
            result = algorithm.run(seed=seed)
            best_costs.append(result['best_cost'])
            convergence_data.append(result['convergence'])
            run_times.append(time.time() - seed_start)
        
        print(f"    Completed 10 runs in {time.time() - start_time:.2f}s")
        all_convergence_data.append(convergence_data)
        
        gaps = [(cost - exact_cost) / exact_cost * 100 for cost in best_costs]
        
        metrics = {
            'instance': instance_name, 'nodes': instance['n_nodes'], 'exact_cost': exact_cost,
            'ennsga_mean_cost': np.mean(best_costs), 'ennsga_std_cost': np.std(best_costs),
            'stat_mean_gap_%': np.mean(gaps), 'stat_std_gap_%': np.std(gaps),
            'stat_min_gap_%': np.min(gaps), 'stat_max_gap_%': np.max(gaps),
            'stat_success_rate_1%': np.sum(np.array(gaps) <= 1.0) / len(gaps) * 100,
            'ennsga_mean_time_s': np.mean(run_times), 'ennsga_std_time_s': np.std(run_times)
        }
        
        results.append(metrics)
        
        print(f"  Generating individual plots for {instance_name}...")
        plot_individual_convergence(instance_name, convergence_data, exact_cost)
        
        print(f"\n  SUMMARY for {instance_name}:")
        print(f"    • Exact cost: ${exact_cost:.2f}")
        print(f"    • Mean En-NSGA-II cost: ${metrics['ennsga_mean_cost']:.2f}")
        print(f"    • Mean optimality gap: {metrics['stat_mean_gap_%']:.2f}%")
        print(f"    • Success rate (≤1% gap): {metrics['stat_success_rate_1%']:.1f}%")
        print(f"    • Mean computation time: {metrics['ennsga_mean_time_s']:.3f}s")
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    summary_cols = ['instance', 'nodes', 'exact_cost', 'ennsga_mean_cost',
                   'stat_mean_gap_%', 'stat_std_gap_%', 'stat_success_rate_1%', 'ennsga_mean_time_s']
    df_summary = df_results[summary_cols].copy()
    print("\n", df_summary.to_string(index=False, float_format=lambda x: f'{x:.2f}' if abs(x) < 100 else f'{x:.0f}'))
    
    print("\nGenerating cross-instance comparison plot...")
    plot_instance_comparison_summary(df_results)
    
    df_results.to_csv('detailed_validation_results.csv', index=False)
    df_summary.to_csv('validation_summary_table.csv', index=False)
    
    latex_table = df_summary.to_latex(index=False, float_format="%.2f")
    with open('validation_table.tex', 'w') as f:
        f.write(latex_table)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nFiles generated:")
    print("  1. convergence_*.png - Individual convergence plots")
    print("  2. instance_comparison_summary.png - Cross-instance comparison")
    print("  3. detailed_validation_results.csv - Complete results")
    print("  4. validation_summary_table.csv - Summary table")
    print("  5. validation_table.tex - LaTeX table")
    
    return df_results, df_summary, all_convergence_data


if __name__ == "__main__":
    print("\n" + "="*80)
    print("INDIVIDUAL INSTANCE VALIDATION SCRIPT")
    print("="*80)
    
    df_results, df_summary, convergence_data = run_validation_with_individual_plots()
    
    print("\n" + "="*80)
    print("KEY FINDINGS FOR PAPER:")
    print("="*80)
    
    for idx, row in df_summary.iterrows():
        print(f"\n{row['instance']}:")
        print(f"  • Optimality Gap: {row['stat_mean_gap_%']:.2f}%")
        print(f"  • Success Rate (≤1% gap): {row['stat_success_rate_1%']:.1f}%")
        print(f"  • Mean Computation Time: {row['ennsga_mean_time_s']:.3f}s")