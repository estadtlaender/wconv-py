from abc import ABC, abstractmethod
from typing import Set, List, Any, Union, Tuple, Callable, Dict, Optional, Hashable, Self, TypeVar, Container, Generic
import random as rnd
import itertools as it
from collections import deque
from heapq import heappush, heappop, heapify

import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx

ThetaType = int | float
Distance = Callable[[int, int], ThetaType]
Interval = Callable[[int, int], set[int]]


class Point(ABC, Hashable):
    @abstractmethod
    def distance(self, other: Self) -> ThetaType:
        return 0

    @abstractmethod
    def singleton(self) -> "Block[Self]":
        raise NotImplementedError()


P = TypeVar('P', bound=Point)


class Block(ABC, Hashable, Container, Generic[P]):
    @abstractmethod
    def distance(self, other: Self) -> ThetaType:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def singleton(cls, p: P) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def membership(self, p: P) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def join(self, other: Self) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def connectivity_index(self) -> ThetaType:
        raise NotImplementedError()

    def __contains__(self, item) -> bool:
        return self.membership(item)


class IntervalFunction:
    def __init__(self, ground_set: Optional[Set[Any]], intervals: Optional[Dict[Any, Dict[Any, Set[Any]]]]):
        self.ground_set = ground_set
        self.intervals = intervals

    def __call__(self, x: Any, y: Any) -> Set[Any]:
        return self.intervals[x][y] if (x in self.intervals and y in self.intervals[x]) else {x, y}


class Metric:
    def __init__(self, ground_set: Optional[Set[Any]], dists: Optional[Dict[Any, Dict[Any, float]]]):
        self.ground_set = ground_set
        self.dists = dists

    def __call__(self, x: Any, y: Any) -> float:
        return self.dists[x][y] if (x in self.dists and y in self.dists[x]) else float('inf')


class EuclideanMetric(Metric):
    def __init__(self, points: np.ndarray):
        self.points = points
        ground_set = set(range(len(points)))
        dists = distance_matrix(points, points)
        super().__init__(ground_set, dists)

    def __call__(self, x: int, y: int) -> float:
        return self.dists[x][y] if (0 <= x < len(self.dists)) and (0 <= y < len(self.dists)) else float('inf')


class BoxInterval(IntervalFunction):
    def __init__(self, points: np.ndarray):
        ground_set = set(range(len(points)))
        intervals = {}
        for x, y in it.combinations_with_replacement(ground_set, r=2):
            bl = np.minimum(points[x], points[y])
            ur = np.maximum(points[x], points[y])
            interval = set(np.flatnonzero(np.all(
                (bl <= points) & (points <= ur), axis=1
            )))
            if x not in intervals:
                intervals[x] = {y: interval}
            else:
                intervals[x][y] = interval
            if y not in intervals:
                intervals[y] = {x: interval}
            else:
                intervals[y][x] = interval
        super().__init__(ground_set, intervals)


class QMSolver:
    def __init__(self,
                 ground_set: Set[int],
                 metric: Callable[[int, int], float],
                 interval: Callable[[int, int], Set[int]]):
        self.ground_set = ground_set
        self.metric = metric
        self.interval = interval

    def quantified_membership(self, points: Set[int], theta: float, dropout_rate: Optional[float] = None, seed: Optional[int] = None):
        rng = None
        if dropout_rate is not None:
            rng = np.random.RandomState(seed=seed)
        closure = set(points)
        levels = {x: 0 for x in points}
        queue = [
            (self.metric(u, v), (min(u, v), max(u, v)))
            for u, v in it.combinations(points, r=2) if self.metric(u, v) <= theta
        ]
        heapify(queue)
        while queue:
            l, (u, v) = heappop(queue)
            if dropout_rate is not None:
                if rng.random() < dropout_rate:
                    continue
            intrvl = self.interval(u, v)
            news = intrvl - closure
            for x in news:
                closure.add(x)
                levels[x] = l
            for x, y in it.product(news, closure):
                if self.metric(x, y) <= theta:
                    heappush(queue, (max(l, self.metric(x, y)), (x, y)))
        return {(x, levels[x]) for x in closure}


class Context:
    ground_set = []
    dists = None

    def __init__(self, ground_set: List[Any], dists: np.ndarray):
        self.ground_set = ground_set
        self.dists = dists

    def __len__(self):
        return len(self.ground_set)

    def map_to_idx(self, elements) -> Set[int]:
        return {self.ground_set.index(el) for el in elements if el in self.ground_set}

    def get(self, idx) -> Any:
        return self.ground_set[idx]

    def dist(self, idx, idy) -> np.float64:
        return self.dists[idx][idy]

    def sample(self, k: int) -> Set[int]:
        return set(rnd.sample(range(len(self)), k))


class ParameterizedContext(Context):
    theta = float('NaN')
    epsilon = float('NaN')

    def __init__(self, ground_set: List[Any], dists: np.ndarray, theta: float = 1.0, epsilon: float = 1.0):
        super().__init__(ground_set, dists)
        self.theta = theta
        self.epsilon = epsilon

    def weakly_convex_hull(self, points: Set[int], loop_count: bool = False, loop_limit: int = -1):
        return weakly_convex_hull(points, self, self.theta, self.epsilon, loop_count=loop_count, loop_limit=loop_limit)


class PrecomputedContext(Context):
    theta = float('NaN')
    epsilon = float('NaN')
    atomics = None
    nbhds = None

    def __init__(self, ground_set: List[Any], dists: np.ndarray, theta: float = 1.0, epsilon: float = 0.0):
        super().__init__(ground_set, dists)
        self.theta, self.epsilon = theta, epsilon
        self._preprocess()

    def _atomic_closure(self, x, y) -> Set[int]:
        closure = {x, y}
        queue = deque([(x, y)])
        while queue:
            a, b = queue.popleft()
            if len(self.nbhds[a]) > len(self.nbhds[b]):
                a, b = b, a
            for c in self.nbhds[a]:
                if ((c not in closure)
                        and (c in self.nbhds[b])
                        and (self.dist(a, c) + self.dist(c, b) <= self.dist(a, b) + self.epsilon)):
                    closure.add(c)
                    queue.extend((d, c) for d in closure if self.dist(d, c) <= self.theta)
        return closure

    def _preprocess(self) -> None:
        self.nbhds = {idx: set() for idx in range(len(self))}
        for x, y in it.combinations(range(len(self)), r=2):
            if self.dist(x, y) <= (self.theta + self.epsilon / 2.0):
                self.nbhds[x].add(y)
                self.nbhds[y].add(x)

        self.atomics = {}
        for x, y in it.combinations(range(len(self)), r=2):
            if self.dist(x, y) <= self.theta:
                witnessed = self._atomic_closure(x, y)
                if x not in self.atomics:
                    self.atomics[x] = {y: witnessed}
                else:
                    self.atomics[x][y] = witnessed

    def weakly_convex_hull(self, points: Set[int]):
        closure = set(points)
        queue = deque([(min(x, y), max(x, y)) for x, y in it.combinations_with_replacement(points, r=2)])
        while queue:
            x, y = queue.popleft()
            if (x in self.atomics) and (y in self.atomics[x]):
                witnessed = self.atomics[x][y]
                for z in witnessed:
                    if z not in closure:
                        closure.add(z)
                        queue.extend((min(w, z), max(w, z)) for w in closure if self.dist(w, z) <= self.theta)
        return closure


def create_euclidean_context(ground_set: np.ndarray, theta: float = None, epsilon: float = None,
                             precompute: bool = False) -> Context:
    dists = distance_matrix(ground_set, ground_set)
    if (theta is not None) and (epsilon is not None):
        if precompute:
            return PrecomputedContext(list(ground_set), dists, theta, epsilon)
        else:
            return ParameterizedContext(list(ground_set), dists, theta, epsilon)
    else:
        # TODO: log a warning if theta or epsilon is None (because it won't be used)
        # TODO: log a warning if precompute is true (because then parameters are required)
        return Context(list(ground_set), dists)


def create_uniform_context(num_points: int, theta: float = None, epsilon: float = None,
                           density: float = float('nan'), precompute: bool = False) -> Context:
    hi = np.nextafter(1.0, 2.0)
    if not np.isnan(density):
        hi = np.nextafter(np.sqrt(num_points / density), np.infty)
    ground_set = np.random.uniform(0.0, hi, size=(num_points, 2))
    return create_euclidean_context(ground_set, theta, epsilon, precompute)


def weakly_convex_hull(points: Set[int], context: Context, theta: float = 1.0, epsilon: float = 0.0,
                       loop_count: bool = False, loop_limit: int = None) -> Union[Set[int], Tuple[Set[int], int]]:
    closure, change = set(points), True
    outer_loop_count = 0
    while change and ((loop_limit is None) or (outer_loop_count < loop_limit)):
        outer_loop_count += 1
        change = False
        for x, y in it.combinations_with_replacement(closure, 2):
            if context.dists[x][y] <= theta:
                for z in range(len(context)):
                    if ((z not in closure) and
                            (context.dists[x][y] + epsilon >= context.dists[x][z] + context.dists[z][y])):
                        closure.add(z)
                        change = True
    if loop_count:
        return closure, outer_loop_count
    else:
        return closure


def __neighborhood(x: int, neighborhood: Dict[int, List[Tuple[int, float]]], radius: float) -> Set[int]:
    ball = set()
    for y, d in neighborhood[x]:
        if d > radius:
            break
        ball.add(y)
    return ball


def __common_neighborhood(x: int, y: int, distances: np.ndarray, neighborhood: np.ndarray, radius: float) -> Set[int]:
    buf = set()
    intersection = set()
    for z in neighborhood[x]:
        d = distances[z, x]
        if d > radius:
            break
        buf.add(z)
    for z in neighborhood[y]:
        d = distances[z, y]
        if d > radius:
            break
        if z in buf:
            intersection.add(z)
    return intersection


def __is_between(x: int, y: int, z: int, epsilon: float, dists: Union[np.ndarray, Dict[int, Dict[int, float]]]) -> bool:
    return dists[x][z] + dists[z][y] <= dists[x][y] + epsilon


def fast_weakly_convex_hull(points: Set[int], distances: Dict[int, Dict[int, float]],
                            neighborhood: Dict[int, List[Tuple[int, float]]], theta: float = 1.0,
                            epsilon: float = 0.0):
    closure = set()
    edges = set()
    queue = deque(points)
    marked = set(points)
    while queue:
        x = queue.pop()
        closure.add(x)
        for y in it.takewhile(lambda u: distances[x, u] <= theta, neighborhood[x]):
            d = distances[x, y]
            if y not in closure:
                continue
            edges.add((x, y))
            for z in __common_neighborhood(x, y, distances, neighborhood, d + epsilon/2):
                if z not in marked and __is_between(x, y, z, epsilon, distances):
                    marked.add(z)
                    queue.appendleft(z)
                    edges.add((x, z))
                    edges.add((z, y))
    g = nx.Graph()
    g.add_nodes_from(closure)
    g.add_edges_from(edges)
    return list(nx.connected_components(g))
