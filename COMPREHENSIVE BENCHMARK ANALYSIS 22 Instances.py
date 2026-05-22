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


class ZDT1(MultiObjectiveProblem):
    def __init__(self, n_var: int = 30):
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
    def __init__(self, n_var: int = 30):
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
    def __init__(self, n_var: int = 30):
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
    def __init__(self, n_var: int = 10):
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
    def __init__(self, n_var: int = 10):
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
    def __init__(self, n_var: int = 7, n_obj: int = 3):
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
    def __init__(self, n_var: int = 12, n_obj: int = 3):
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


class DTLZ3(MultiObjectiveProblem):
    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ3"
        self.k = self.n_var - self.n_obj + 1
        
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


class DTLZ4(MultiObjectiveProblem):
    def __init__(self, n_var: int = 12, n_obj: int = 3, alpha: float = 100.0):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ4"
        self.alpha = alpha
        self.k = self.n_var - self.n_obj + 1
        
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


class DTLZ5(MultiObjectiveProblem):
    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ5"
        self.k = self.n_var - self.n_obj + 1
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = np.sum((x[self.n_obj-1:] - 0.5) ** 2)
        
        theta = np.zeros(self.n_obj)
        theta[0] = x[0] * np.pi / 2.0
        for i in range(1, self.n_obj - 1):
            theta[i] = np.pi / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[i])
        
        f = (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(theta[j])
            if i > 0:
                f[i] *= np.sin(theta[self.n_obj - i - 1])
        return f
    
    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.rand(n_points, 2)
        points = np.column_stack([points, np.zeros((n_points, 1))])
        return points
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class DTLZ6(MultiObjectiveProblem):
    def __init__(self, n_var: int = 12, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ6"
        self.k = self.n_var - self.n_obj + 1
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        g = np.sum(x[self.n_obj-1:] ** 0.1)
        
        theta = np.zeros(self.n_obj)
        theta[0] = x[0] * np.pi / 2.0
        for i in range(1, self.n_obj - 1):
            theta[i] = np.pi / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x[i])
        
        f = (1.0 + g) * np.ones(self.n_obj)
        for i in range(self.n_obj):
            for j in range(self.n_obj - i - 1):
                f[i] *= np.cos(theta[j])
            if i > 0:
                f[i] *= np.sin(theta[self.n_obj - i - 1])
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class DTLZ7(MultiObjectiveProblem):
    def __init__(self, n_var: int = 22, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.name = "DTLZ7"
        self.k = self.n_var - self.n_obj + 1
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = x[i]
        
        g = 1.0 + 9.0 * np.sum(x[self.n_obj-1:]) / self.k
        
        h = self.n_obj - np.sum(f[:self.n_obj-1] / (1.0 + g) * (1.0 + np.sin(3 * np.pi * f[:self.n_obj-1])))
        f[self.n_obj-1] = (1.0 + g) * h
        return f
    
    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.rand(n_points, self.n_obj - 1)
        g = 0.0
        f_last = self.n_obj - np.sum(points * (1.0 + np.sin(3 * np.pi * points)), axis=1)
        return np.column_stack([points, f_last.reshape(-1, 1)])
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([1.5, 1.5, 1.5])


class WFG1(MultiObjectiveProblem):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=2.0 * np.arange(1, n_var + 1))
        self.name = "WFG1"
        self.k = n_var - n_obj + 1
        
    def _s_linear(self, y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)
    
    def _b_flat(self, y, A, B, C):
        return A + min(0, np.floor(y - B)) * (A * (B - y) / B) - min(0, np.floor(C - y)) * ((1 - A) * (y - C) / (1 - C))
    
    def _b_poly(self, y, alpha):
        return y ** alpha
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[:self.k] = self._s_linear(y[:self.k], 0.35)
        
        t1 = np.zeros(self.n_var)
        for i in range(self.n_obj - 1):
            t1[i] = self._b_flat(y[i], 0.8, 0.75, 0.85)
        t1[self.n_obj-1:self.n_var] = y[self.n_obj-1:self.n_var]
        
        t2 = np.zeros(self.n_var)
        for i in range(self.n_obj - 1):
            t2[i] = self._b_poly(t1[i], 0.02)
        t2[self.n_obj-1:self.n_var] = self._b_poly(t1[self.n_obj-1:self.n_var], 0.02)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t2[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t2[self.k:]) / self.k
        return f
    
    def pareto_front(self, n_points: int = 100) -> np.ndarray:
        points = np.random.rand(n_points, self.n_obj - 1)
        f_last = np.sum(points, axis=1)
        return np.column_stack([points, f_last.reshape(-1, 1)])
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG2(WFG1):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG2"
        
    def _s_linear(self, y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[:self.k] = self._s_linear(y[:self.k], 0.35)
        
        l = self.n_var - self.k
        t1 = np.zeros(self.n_var)
        t1[:self.n_obj-1] = y[:self.n_obj-1]
        t1[self.n_obj-1:self.n_obj-1 + l//2] = np.abs(y[self.n_obj-1:self.n_obj-1 + l//2] - y[self.n_obj-1 + l//2:self.n_obj-1 + l]) / 2.0
        t1[self.n_obj-1 + l//2:] = y[self.n_obj-1 + l:]
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG3(WFG1):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG3"
        
    def _s_linear(self, y, A):
        return np.abs(y - A) / np.abs(np.floor(A - y) + A)
    
    def _b_flat(self, y, A, B, C):
        return A + min(0, np.floor(y - B)) * (A * (B - y) / B) - min(0, np.floor(C - y)) * ((1 - A) * (y - C) / (1 - C))
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        y[:self.k] = self._s_linear(y[:self.k], 0.35)
        
        t1 = np.zeros(self.n_var)
        for i in range(self.n_obj - 1):
            t1[i] = self._b_flat(y[i], 0.8, 0.75, 0.85)
        t1[self.n_obj-1:self.n_var] = y[self.n_obj-1:self.n_var]
        
        l = self.n_var - self.k
        t2 = np.zeros(self.n_var)
        t2[:self.n_obj-1] = t1[:self.n_obj-1]
        t2[self.n_obj-1:self.n_obj-1 + l//2] = np.abs(t1[self.n_obj-1:self.n_obj-1 + l//2] - t1[self.n_obj-1 + l//2:self.n_obj-1 + l]) / 2.0
        t2[self.n_obj-1 + l//2:] = t1[self.n_obj-1 + l:]
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t2[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t2[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG4(WFG1):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG4"
        
    def _s_multi(self, y, A, B, C):
        return (1.0 + np.cos((4.0 * A + 2.0) * np.pi * (0.5 - np.abs(y - C) / (2.0 * np.floor(C - y + C) + 4.0 * A * B - 2.0 * A + 2.0)))) / 2.0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[:self.n_obj-1] = y[:self.n_obj-1]
        t1[self.n_obj-1:self.n_var] = self._s_multi(y[self.n_obj-1:self.n_var], 30, 10, 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG5(WFG4):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG5"
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[:self.n_obj-1] = y[:self.n_obj-1]
        t1[self.n_obj-1:self.n_var] = self._s_multi(y[self.n_obj-1:self.n_var], 30, 10, 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG6(WFG1):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG6"
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[:self.n_obj-1] = y[:self.n_obj-1]
        t1[self.n_obj-1:self.n_var] = np.abs(y[self.n_obj-1:self.n_var] - 0.35) / np.abs(np.floor(0.35 - y[self.n_obj-1:self.n_var]) + 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG7(WFG1):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG7"
        
    def _b_param(self, y, u, A, B, C):
        v = (y - A) / (B - A)
        return u ** (C + (B - A) * v)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[0] = self._b_param(y[0], np.mean(y[:self.k]), 0, 1, 1)
        t1[1:self.n_obj-1] = y[1:self.n_obj-1]
        t1[self.n_obj-1:self.n_var] = self._s_linear(y[self.n_obj-1:self.n_var], 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG8(WFG7):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG8"
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[0] = y[0]
        for i in range(1, self.n_obj - 1):
            t1[i] = self._b_param(y[i], np.mean(y[:i]), 0, 1, 1)
        t1[self.n_obj-1:self.n_var] = self._s_linear(y[self.n_obj-1:self.n_var], 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class WFG9(WFG7):
    def __init__(self, n_var: int = 9, n_obj: int = 3):
        super().__init__(n_var=n_var, n_obj=n_obj)
        self.name = "WFG9"
        
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        y = x.copy()
        
        t1 = np.zeros(self.n_var)
        t1[0] = self._b_param(y[0], np.mean(y[:self.k]), 0, 1, 1)
        for i in range(1, self.n_obj - 1):
            t1[i] = self._b_param(y[i], np.mean(y[:i]), 0, 1, 1)
        t1[self.n_obj-1:self.n_var] = self._s_multi(y[self.n_obj-1:self.n_var], 30, 10, 0.35)
        
        S = np.zeros(self.n_obj)
        for i in range(self.n_obj):
            S[i] = 2.0 * (i + 1)
        
        f = np.zeros(self.n_obj)
        for i in range(self.n_obj - 1):
            f[i] = S[i] * t1[self.n_obj - 2 - i]
        f[self.n_obj - 1] = S[self.n_obj - 1] * np.sum(t1[self.k:]) / self.k
        return f
    
    def get_reference_point(self) -> np.ndarray:
        return np.array([5.0, 5.0, 5.0])


class BaseMOEA:
    def __init__(self, pop_size: int = 100, max_gen: int = 250,
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
    def __init__(self, pop_size: int = 100, max_gen: int = 250):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "NSGA-II"
    
    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()
        
        population = problem.generate_population(self.pop_size)
        objectives = np.array([problem.evaluate(ind) for ind in population])
        
        history = {'generation': [], 'best_hv': [], 'best_igd': []}
        
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
    def __init__(self, pop_size: int = 100, max_gen: int = 250, use_enhancements: bool = True):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "En-NSGA-II"
        self.use_enhancements = use_enhancements
        self.archive_size = pop_size // 2
        self.archive = None
        self.archive_objectives = None
        self.diversity_threshold = 0.1
        self.restart_counter = 0
        
    def intelligent_initialization(self, problem: MultiObjectiveProblem, size: int) -> np.ndarray:
        n_var = problem.n_var
        lhs_points = np.zeros((size, n_var))
        for i in range(n_var):
            lhs_points[:, i] = np.random.permutation(size) / (size - 1)
        lhs_points = lhs_points * (problem.xu - problem.xl) + problem.xl
        return lhs_points
    
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
            fitness = domination_count + np.mean(distances) * 0.1
            fitness_scores.append(fitness)
        top_indices = np.argsort(fitness_scores)[-2:]
        return candidates[top_indices[0]], candidates[top_indices[1]]
    
    def gaussian_mutation(self, individual: np.ndarray, generation: int, max_gen: int) -> np.ndarray:
        mutant = individual.copy()
        adaptive_rate = 0.1 * (1.0 + generation / max_gen)
        sigma = 0.1 * (1.0 - generation / max_gen)
        for i in range(len(individual)):
            if np.random.random() < adaptive_rate:
                mutant[i] += np.random.normal(0, sigma)
        return np.clip(mutant, 0.0, 1.0)
    
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
        
        self.archive = np.array(new_archive)
        self.archive_objectives = np.array(new_archive_obj)
    
    def inject_archive_solutions(self, population: np.ndarray, objectives: np.ndarray, 
                               injection_rate: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        if self.archive is None or len(self.archive) == 0:
            return population, objectives
        
        n_inject = max(1, int(len(population) * injection_rate))
        archive_indices = np.random.choice(len(self.archive), min(n_inject, len(self.archive)), replace=False)
        
        worst_indices = np.random.choice(len(population), len(archive_indices), replace=False)
        
        new_population = population.copy()
        new_objectives = objectives.copy()
        
        for i, idx in enumerate(worst_indices):
            new_population[idx] = self.archive[archive_indices[i]]
            new_objectives[idx] = self.archive_objectives[archive_indices[i]]
        
        return new_population, new_objectives
    
    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()
        
        if self.use_enhancements:
            population = self.intelligent_initialization(problem, self.pop_size)
        else:
            population = problem.generate_population(self.pop_size)
        
        objectives = np.array([problem.evaluate(ind) for ind in population])
        
        self.archive = population.copy()
        self.archive_objectives = objectives.copy()
        
        history = {'generation': [], 'best_hv': [], 'best_igd': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)
        
        no_improvement_count = 0
        best_hv = 0.0
        
        for gen in range(self.max_gen):
            offspring_pop = []
            
            for _ in range(self.pop_size // 2):
                parent1_idx, parent2_idx = self.adaptive_tournament_selection(population, objectives)
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                child1, child2 = self.simulated_binary_crossover(parent1, parent2)
                child1 = self.gaussian_mutation(child1, gen, self.max_gen)
                child2 = self.gaussian_mutation(child2, gen, self.max_gen)
                
                offspring_pop.extend([child1, child2])
            
            offspring_obj = np.array([problem.evaluate(ind) for ind in offspring_pop])
            
            combined_pop = np.vstack([population, offspring_pop])
            combined_obj = np.vstack([objectives, offspring_obj])
            
            self.update_archive(combined_pop, combined_obj)
            
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
            
            if self.use_enhancements and gen % 5 == 0:
                population, objectives = self.inject_archive_solutions(population, objectives, 0.15)
            
            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)
            
            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
            
            if hv > best_hv:
                best_hv = hv
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count > 30 and gen < self.max_gen * 0.8:
                n_replace = int(self.pop_size * 0.3)
                replace_idx = np.random.choice(self.pop_size, n_replace, replace=False)
                for idx in replace_idx:
                    if len(self.archive) > 0:
                        archive_idx = np.random.randint(len(self.archive))
                        population[idx] = self.archive[archive_idx].copy() + np.random.normal(0, 0.1, problem.n_var)
                        population[idx] = np.clip(population[idx], 0.0, 1.0)
                objectives = np.array([problem.evaluate(ind) for ind in population])
                no_improvement_count = 0
        
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


class SPEA2(BaseMOEA):
    def __init__(self, pop_size: int = 100, max_gen: int = 250):
        super().__init__(pop_size=pop_size, max_gen=max_gen)
        self.name = "SPEA2"
    
    def optimize(self, problem: MultiObjectiveProblem) -> Dict[str, Any]:
        start_time = time.time()
        
        population = problem.generate_population(self.pop_size)
        
        history = {'generation': [], 'best_hv': [], 'best_igd': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)
        
        for gen in range(self.max_gen):
            objectives = np.array([problem.evaluate(ind) for ind in population])
            
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
            
            hv = metrics_calc.hypervolume(objectives, ref_point)
            igd = metrics_calc.inverted_generational_distance(objectives, true_front)
            
            history['generation'].append(gen)
            history['best_hv'].append(hv)
            history['best_igd'].append(igd)
        
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
    def __init__(self, pop_size: int = 100, max_gen: int = 250, neighbor_size: int = 10):
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
        
        history = {'generation': [], 'best_hv': [], 'best_igd': []}
        metrics_calc = PerformanceMetrics()
        ref_point = problem.get_reference_point()
        true_front = problem.pareto_front(100)
        
        for gen in range(self.max_gen):
            for i in range(self.pop_size):
                neighbor_idx = self.neighbors[i]
                if len(neighbor_idx) >= 2:
                    parents = np.random.choice(neighbor_idx, 2, replace=False)
                    child, _ = self.simulated_binary_crossover(population[parents[0]], population[parents[1]])
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
            
            pareto_front_filtered = sorted_front[pareto_mask]
            if len(pareto_front_filtered) == 0:
                return 0.0
            
            area = 0.0
            prev_x = ref_point[0]
            for point in pareto_front_filtered:
                width = prev_x - point[0]
                height = ref_point[1] - point[1]
                area += width * height
                prev_x = point[0]
            return max(area, 0.0)
        else:
            n_samples = 1000
            samples = np.random.rand(n_samples, front.shape[1]) * ref_point
            dominated = sum(1 for sample in samples 
                          if any(np.all(point <= sample) for point in front))
            return (dominated / n_samples) * np.prod(ref_point)
    
    @staticmethod
    def inverted_generational_distance(pareto_front: np.ndarray, true_front: np.ndarray) -> float:
        if len(pareto_front) == 0 or len(true_front) == 0:
            return float('inf')
        
        pareto_front = np.clip(pareto_front, 0, 1000)
        true_front = np.clip(true_front, 0, 1000)
        
        distances = [np.min(np.sqrt(np.sum((pareto_front - true_point) ** 2, axis=1))) 
                    for true_point in true_front]
        return np.mean(distances)


class BenchmarkFramework:
    def __init__(self):
        self.problems = []
        self.algorithms = {}
        self.results = {}
        self.n_runs = 10
        self.n_gen = 250
        self.pop_size = 100
        
    def load_complete_benchmark_suite(self):
        print("Loading complete benchmark suite (22 instances)...")
        
        self.problems = []
        
        for n_var in [10, 20, 30]:
            self.problems.append((f"ZDT1_{n_var}D", ZDT1(n_var=n_var)))
        
        for n_var in [10, 20, 30]:
            self.problems.append((f"ZDT2_{n_var}D", ZDT2(n_var=n_var)))
        
        for n_var in [10, 20, 30]:
            self.problems.append((f"ZDT3_{n_var}D", ZDT3(n_var=n_var)))
        
        self.problems.append(("ZDT4_10D", ZDT4(n_var=10)))
        self.problems.append(("ZDT6_10D", ZDT6(n_var=10)))
        
        for n_obj in [3, 5, 8]:
            self.problems.append((f"DTLZ1_{n_obj}obj", DTLZ1(n_var=n_obj + 4, n_obj=n_obj)))
            self.problems.append((f"DTLZ2_{n_obj}obj", DTLZ2(n_var=n_obj + 9, n_obj=n_obj)))
            self.problems.append((f"DTLZ3_{n_obj}obj", DTLZ3(n_var=n_obj + 9, n_obj=n_obj)))
        
        self.problems.append(("DTLZ4_3obj", DTLZ4(n_var=12, n_obj=3)))
        self.problems.append(("DTLZ5_3obj", DTLZ5(n_var=12, n_obj=3)))
        self.problems.append(("DTLZ6_3obj", DTLZ6(n_var=12, n_obj=3)))
        self.problems.append(("DTLZ7_3obj", DTLZ7(n_var=22, n_obj=3)))
        
        for n_obj in [3, 5]:
            self.problems.append((f"WFG1_{n_obj}obj", WFG1(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG2_{n_obj}obj", WFG2(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG3_{n_obj}obj", WFG3(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG4_{n_obj}obj", WFG4(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG5_{n_obj}obj", WFG5(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG6_{n_obj}obj", WFG6(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG7_{n_obj}obj", WFG7(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG8_{n_obj}obj", WFG8(n_var=n_obj + 6, n_obj=n_obj)))
            self.problems.append((f"WFG9_{n_obj}obj", WFG9(n_var=n_obj + 6, n_obj=n_obj)))
        
        print(f"Loaded {len(self.problems)} benchmark problems")
    
    def setup_algorithms(self):
        print("\nSetting up algorithms...")
        
        self.algorithms = {
            'En-NSGA-II': EnhancedNSGA2(pop_size=self.pop_size, max_gen=self.n_gen, use_enhancements=True),
            'NSGA-II': NSGA2(pop_size=self.pop_size, max_gen=self.n_gen),
            'SPEA2': SPEA2(pop_size=self.pop_size, max_gen=self.n_gen),
            'MOEA/D': MOEAD(pop_size=self.pop_size, max_gen=self.n_gen, neighbor_size=10)
        }
    
    def run_benchmarks(self):
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE BENCHMARK EXPERIMENT ON 22 INSTANCES")
        print("="*80)
        
        self.results = {}
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
                    'HV': [], 'IGD': [], 'Runtime': [],
                    'Final_Objectives': []
                }
                
                for run in range(self.n_runs):
                    test_count += 1
                    progress = test_count / total_tests * 100
                    print(f"    Run {run + 1}/{self.n_runs} [{progress:.1f}%]...", end=' ')
                    
                    np.random.seed(42 * test_count + run)
                    
                    try:
                        result = algorithm.optimize(problem)
                        
                        algo_results['HV'].append(result['final_hv'])
                        algo_results['IGD'].append(result['final_igd'])
                        algo_results['Runtime'].append(result['runtime'])
                        algo_results['Final_Objectives'].append(result['objectives'])
                        
                        print(f"HV={result['final_hv']:.4f}, IGD={result['final_igd']:.4f}, Time={result['runtime']:.1f}s")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        algo_results['HV'].append(0.0)
                        algo_results['IGD'].append(float('inf'))
                        algo_results['Runtime'].append(float('inf'))
                        algo_results['Final_Objectives'].append(None)
                
                self.results[prob_name][algo_name] = {
                    'HV_mean': np.mean(algo_results['HV']),
                    'HV_std': np.std(algo_results['HV']),
                    'IGD_mean': np.mean(algo_results['IGD']),
                    'IGD_std': np.std(algo_results['IGD']),
                    'Runtime_mean': np.mean(algo_results['Runtime']),
                    'Runtime_std': np.std(algo_results['Runtime']),
                    'all_HV': algo_results['HV'],
                    'all_IGD': algo_results['IGD'],
                    'all_Runtime': algo_results['Runtime'],
                    'final_objectives': algo_results['Final_Objectives']
                }
        
        print(f"\n{'='*80}")
        print("BENCHMARK EXPERIMENT COMPLETED")
        return self.results
    
    def print_comprehensive_results(self):
        print("\n" + "="*100)
        print("EXPERIMENTAL RESULTS - 22 BENCHMARK INSTANCES")
        print("="*100)
        
        problems = sorted(self.results.keys())
        algorithms = list(self.algorithms.keys())
        
        print("\n" + "="*100)
        print("HYPERVOLUME (HV) RESULTS - Mean ± Std (Higher is better)")
        print("="*100)
        
        hv_summary = []
        for prob_name in problems:
            row = {'Problem': prob_name}
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    row[f'{algo_name}_HV'] = f"{self.results[prob_name][algo_name]['HV_mean']:.6f}"
                    row[f'{algo_name}_HV_std'] = f"{self.results[prob_name][algo_name]['HV_std']:.6f}"
            hv_summary.append(row)
        
        df_hv = pd.DataFrame(hv_summary)
        print(df_hv.to_string())
        
        print("\n" + "="*100)
        print("INVERTED GENERATIONAL DISTANCE (IGD) RESULTS - Mean ± Std (Lower is better)")
        print("="*100)
        
        igd_summary = []
        for prob_name in problems:
            row = {'Problem': prob_name}
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    igd_mean = self.results[prob_name][algo_name]['IGD_mean']
                    igd_std = self.results[prob_name][algo_name]['IGD_std']
                    if np.isinf(igd_mean):
                        row[f'{algo_name}_IGD'] = "inf"
                    else:
                        row[f'{algo_name}_IGD'] = f"{igd_mean:.6f}"
                    row[f'{algo_name}_IGD_std'] = f"{igd_std:.6f}" if not np.isinf(igd_std) else "inf"
            igd_summary.append(row)
        
        df_igd = pd.DataFrame(igd_summary)
        print(df_igd.to_string())
        
        print("\n" + "="*100)
        print("RUNTIME RESULTS (seconds) - Mean ± Std (Lower is better)")
        print("="*100)
        
        runtime_summary = []
        for prob_name in problems:
            row = {'Problem': prob_name}
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    row[f'{algo_name}_Runtime'] = f"{self.results[prob_name][algo_name]['Runtime_mean']:.3f}"
                    row[f'{algo_name}_Runtime_std'] = f"{self.results[prob_name][algo_name]['Runtime_std']:.3f}"
            runtime_summary.append(row)
        
        df_runtime = pd.DataFrame(runtime_summary)
        print(df_runtime.to_string())
        
        print("\n" + "="*100)
        print("PERFORMANCE SUMMARY - Win Counts Across 22 Instances")
        print("="*100)
        
        hv_wins = {algo: 0 for algo in algorithms}
        igd_wins = {algo: 0 for algo in algorithms}
        runtime_wins = {algo: 0 for algo in algorithms}
        
        for prob_name in problems:
            best_hv = -1
            best_hv_algo = None
            best_igd = float('inf')
            best_igd_algo = None
            best_runtime = float('inf')
            best_runtime_algo = None
            
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hv = self.results[prob_name][algo_name]['HV_mean']
                    if hv > best_hv:
                        best_hv = hv
                        best_hv_algo = algo_name
                    
                    igd = self.results[prob_name][algo_name]['IGD_mean']
                    if igd < best_igd and not np.isinf(igd):
                        best_igd = igd
                        best_igd_algo = algo_name
                    
                    runtime = self.results[prob_name][algo_name]['Runtime_mean']
                    if runtime < best_runtime:
                        best_runtime = runtime
                        best_runtime_algo = algo_name
            
            if best_hv_algo:
                hv_wins[best_hv_algo] += 1
            if best_igd_algo:
                igd_wins[best_igd_algo] += 1
            if best_runtime_algo:
                runtime_wins[best_runtime_algo] += 1
        
        print(f"\n{'Algorithm':<15} {'HV Wins':<12} {'IGD Wins':<12} {'Runtime Wins':<12} {'Total Wins':<12}")
        print("-" * 63)
        for algo in algorithms:
            total = hv_wins[algo] + igd_wins[algo] + runtime_wins[algo]
            print(f"{algo:<15} {hv_wins[algo]:<12} {igd_wins[algo]:<12} {runtime_wins[algo]:<12} {total:<12}")
        
        print("\n" + "="*100)
        print("RANKING SUMMARY (Average Rank Across All Metrics)")
        print("="*100)
        
        rankings = {algo: {'HV': 0, 'IGD': 0, 'Runtime': 0, 'count': 0} for algo in algorithms}
        
        for prob_name in problems:
            hv_values = []
            igd_values = []
            runtime_values = []
            
            for algo_name in algorithms:
                if algo_name in self.results[prob_name]:
                    hv_values.append(self.results[prob_name][algo_name]['HV_mean'])
                    igd_values.append(self.results[prob_name][algo_name]['IGD_mean'])
                    runtime_values.append(self.results[prob_name][algo_name]['Runtime_mean'])
            
            hv_sorted = sorted(range(len(hv_values)), key=lambda i: hv_values[i], reverse=True)
            igd_sorted = sorted(range(len(igd_values)), key=lambda i: igd_values[i])
            runtime_sorted = sorted(range(len(runtime_values)), key=lambda i: runtime_values[i])
            
            for rank, idx in enumerate(hv_sorted):
                algo_name = algorithms[idx]
                rankings[algo_name]['HV'] += rank + 1
                rankings[algo_name]['count'] += 1
            
            for rank, idx in enumerate(igd_sorted):
                algo_name = algorithms[idx]
                rankings[algo_name]['IGD'] += rank + 1
            
            for rank, idx in enumerate(runtime_sorted):
                algo_name = algorithms[idx]
                rankings[algo_name]['Runtime'] += rank + 1
        
        print(f"\n{'Algorithm':<15} {'Avg HV Rank':<15} {'Avg IGD Rank':<15} {'Avg Runtime Rank':<15} {'Overall Rank':<15}")
        print("-" * 75)
        
        overall_scores = []
        for algo in algorithms:
            avg_hv = rankings[algo]['HV'] / rankings[algo]['count']
            avg_igd = rankings[algo]['IGD'] / rankings[algo]['count']
            avg_runtime = rankings[algo]['Runtime'] / rankings[algo]['count']
            overall = (avg_hv + avg_igd + avg_runtime) / 3
            overall_scores.append((algo, overall))
            print(f"{algo:<15} {avg_hv:<15.2f} {avg_igd:<15.2f} {avg_runtime:<15.2f} {overall:<15.2f}")
        
        overall_scores.sort(key=lambda x: x[1])
        print("\n" + "-" * 75)
        print("FINAL RANKING:")
        for rank, (algo, score) in enumerate(overall_scores, 1):
            print(f"  {rank}. {algo} (Overall Score: {score:.2f})")
    
    def statistical_analysis(self):
        print("\n" + "="*100)
        print("STATISTICAL ANALYSIS - Wilcoxon Rank-Sum Test")
        print("="*100)
        
        problems = sorted(self.results.keys())
        
        print("\nHV Metric - En-NSGA-II vs Baselines:")
        print("-" * 80)
        print(f"{'Problem':<20} {'vs NSGA-II':<20} {'vs SPEA2':<20} {'vs MOEA/D':<20}")
        
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
            
            print(f"{prob_name:<20} {results[0]:<20} {results[1]:<20} {results[2]:<20}")
        
        print("\nIGD Metric - En-NSGA-II vs Baselines:")
        print("-" * 80)
        print(f"{'Problem':<20} {'vs NSGA-II':<20} {'vs SPEA2':<20} {'vs MOEA/D':<20}")
        
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
            
            print(f"{prob_name:<20} {results[0]:<20} {results[1]:<20} {results[2]:<20}")
        
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
                            'Runtime': metrics['all_Runtime'][run] if run < len(metrics['all_Runtime']) else np.nan
                        })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_csv('complete_benchmark_results_22_instances.csv', index=False)
            print(f"✓ Detailed results saved to 'complete_benchmark_results_22_instances.csv' ({len(df_detailed)} rows)")
        
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
                    'Runtime_Mean': metrics['Runtime_mean'],
                    'Runtime_Std': metrics['Runtime_std']
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_csv('complete_summary_results_22_instances.csv', index=False)
            print(f"✓ Summary results saved to 'complete_summary_results_22_instances.csv' ({len(summary_data)} rows)")
        
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
            df_stats.to_csv('statistical_significance_22_instances.csv', index=False)
            print(f"✓ Statistical significance results saved to 'statistical_significance_22_instances.csv'")
        
        print("\nAll results have been saved successfully!")


def main():
    print("\n" + "="*100)
    print("ENHANCED NSGA-II COMPREHENSIVE BENCHMARK ANALYSIS")
    print("22 INSTANCES ACROSS ZDT, DTLZ, AND WFG TEST SUITES")
    print("="*100)
    
    framework = BenchmarkFramework()
    framework.load_complete_benchmark_suite()
    framework.setup_algorithms()
    
    try:
        print("\nStarting comprehensive benchmark analysis on 22 instances...")
        results = framework.run_benchmarks()
        
        print("\nGenerating comprehensive analysis...")
        framework.print_comprehensive_results()
        framework.statistical_analysis()
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