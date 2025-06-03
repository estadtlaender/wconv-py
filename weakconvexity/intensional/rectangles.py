from dataclasses import dataclass
from collections.abc import Sequence
from typing import Self, Iterable

import numpy as np

from weakconvexity import Point, Block, ThetaType
import weakconvexity.intensional as intensional


_tau: float = 1e-10
_distance_order: float | int = 1


class Vec(Point, Sequence[float]):
    def __init__(self, coordinates: Iterable[float]):
        self.coordinates = tuple(coordinates)

    def distance(self, other: Self) -> ThetaType:
        return sum(abs(a - b)**_distance_order for (a, b) in zip(self, other, strict=True))**(1/_distance_order)

    def singleton(self) -> "Rectangle":
        return Rectangle.singleton(self)

    def __hash__(self):
        return hash(self.coordinates)

    def __eq__(self, other):
        return isinstance(other, Vec) and self.coordinates == other.coordinates

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    def __repr__(self):
        return f"Vec{list(self)}"


class Rectangle(Block[Vec], Sequence[tuple[float, float]]):
    def __init__(self, first: Iterable[float], second: Iterable[float]):
        lower, upper = zip(*tuple((min(a, b), max(a, b)) for a, b in zip(first, second, strict=True)))
        self.lower_bounds = Vec(lower)
        self.upper_bounds = Vec(upper)

    def distance(self, other: Self) -> ThetaType:
        total = 0.0
        for (lb1, ub1), (lb2, ub2) in zip(self, other, strict=True):
            if ub1 < lb2:
                # no overlap in this dimension, and first interval left from second
                total += (lb2 - ub1)**_distance_order
            elif ub2 < lb1:
                # no overlap in this dimension, and second interval left from first
                total += (lb1 - ub2)**_distance_order

        return total**(1.0/_distance_order)

    @classmethod
    def singleton(cls, p: Vec) -> Self:
        return Rectangle(p, p)

    def membership(self, p: Vec) -> bool:
        return all(
            lb <= coord <= ub
            for lb, coord, ub in zip(self.lower_bounds, p, self.upper_bounds)
        )

    def join(self, other: Self) -> Self:
        lower = Vec(tuple(min(a, b) for a, b in zip(self.lower_bounds, other.lower_bounds)))
        upper = Vec(tuple(max(a, b) for a, b in zip(self.upper_bounds, other.upper_bounds)))
        return Rectangle(lower, upper)

    def connectivity_index(self) -> ThetaType:
        return _tau

    def __hash__(self):
        return hash((self.lower_bounds, self.upper_bounds))

    def __eq__(self, other):
        return (isinstance(other, Rectangle) and (self.lower_bounds == other.lower_bounds)
                and (self.upper_bounds == other.upper_bounds))

    def __len__(self):
        return len(self.lower_bounds)

    def __getitem__(self, index):
        return self.lower_bounds[index], self.upper_bounds[index]

    def __repr__(self):
        return f"Rectangle{list(self)}"


def wconv(theta: float, generators: Iterable[Vec]) -> frozenset[Rectangle]:
    return intensional.wconv(theta, generators)


def chf(positives: Iterable[Vec], negatives: Iterable[Vec], tau: float = _tau) -> None | frozenset[Rectangle]:
    return intensional.chf(positives, negatives, tau=tau)


def heuristic_chf(positives: Iterable[Vec], negatives: Iterable[Vec], tau: float = _tau) -> None | frozenset[tuple[Rectangle, float]]:
    return intensional.heuristic_chf(positives, negatives, tau=tau)


def classification(X: Iterable[Vec], y: Iterable[int], tau: float = _tau) -> None | frozenset[tuple[Rectangle, float, int]]:
    return intensional.classification(X, y, tau=tau)


def classification_from_boxes(inboxes: frozenset[tuple[Rectangle, float, int]], tau: float = _tau) -> None | frozenset[tuple[Rectangle, float, int]]:
    return intensional.classification_from_boxes(inboxes, tau) 


def classification_randomized(
        X: Iterable[Vec], y: Iterable[int], tau: float = _tau,
        rng: None | int | np.random.Generator = None
) -> None | frozenset[tuple[Rectangle, float, int]]:
    return intensional.classification_randomized(X, y, tau=tau, rng=rng)

def postprocess_inflate(
        rectangles: frozenset[tuple[Rectangle], float, int], factor: float = 0.499
) -> frozenset[tuple[Rectangle, float, int]]:
    new_rectangles = set() 

    for i, (rect, tau, label) in enumerate(rectangles): 
        min_dist = min(
            rect.distance(other) 
            for (other, _, other_label) in rectangles if label != other_label
        )
        bounds = [(lb - factor*min_dist, ub + factor*min_dist) for lb, ub in rect]
        lbs, ubs = zip(*bounds)
        new_rect = Rectangle(lbs, ubs)
        new_rectangles.add((new_rect, tau, label))

    return frozenset(new_rectangles)

def postprocess_extend(
        rectangles: frozenset[tuple[Rectangle], float, int], X: Iterable[Vec], y: Iterable[int]
) -> frozenset[tuple[Rectangle, float, int]]: 
    new_rectangles = set(rectangles)
    for rect, theta, label in rectangles: 
        new_rectangle = rect
        changed = False 
        for p, l in zip(X, y): 
            if label == l: 
                candidate = new_rectangle.join(p.singleton())
                reject = False 
                for other, other_theta, other_label in new_rectangles: 
                    if other_label != label and candidate.distance(other) <= max(theta, other_theta): 
                        reject = True
                        break 
                if not reject: 
                    new_rectangle = candidate
                    changed = True 
        if changed: 
            new_rectangles.remove((rect, theta, label))
            new_rectangles.add((new_rectangle, theta, label))
    return frozenset(new_rectangles)

def convert_rectangle_to_rule(rect: Rectangle, features: list[int] | None = None) -> list[tuple[int, tuple[float, float]]]: 
    return [(i if features is None else features[i], (lb, ub)) for i, (lb, ub) in enumerate(rect)]

def convert_lcbs_to_rules(
        n_classes: int, lcbs: frozenset[tuple[Rectangle, float, int]], features: list[int] | None = None 
) -> list[list[list[tuple[int, tuple[float, float]]]]]:
    all_rules = [] 
    for c in range(n_classes): 
        class_c = [convert_rectangle_to_rule(rect, features) for rect, _, label in lcbs if label == c]
        all_rules.append(class_c)
    return all_rules
