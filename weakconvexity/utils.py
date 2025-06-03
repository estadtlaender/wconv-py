import heapq
import pathlib
from typing import Tuple, Union, Any, Dict, Set, Iterable, Container

import networkx as nx
import numpy as np


class MinPriorityQueue:
    REMOVED = None

    def __init__(self):
        self.heap = []
        self.entries = {}

    def change_priority(self, item, priority):
        entry = self.entries.pop(item)
        entry[-1] = MinPriorityQueue.REMOVED
        self.push(item, priority)

    def push(self, item, priority):
        if item in self.entries:
            self.change_priority(item, priority)
        else:
            entry = [priority, item]
            self.entries[item] = entry
            heapq.heappush(self.heap, entry)

    def extract_min(self) -> Tuple[Union[int, float], Any]:
        while self.heap:
            priority, item = heapq.heappop(self.heap)
            if item is not MinPriorityQueue.REMOVED:
                del self.entries[item]
                return priority, item
        raise KeyError('extract_min on empty heap')

    def __contains__(self, item) -> float:
        return item in self.entries

    def __len__(self) -> int:
        return len(self.entries)


class ParameterGrid:
    pass


def summary_graph(g: nx.Graph) -> Dict[str, Any]:
    degrees = np.array([g.degree[v] for v in g])
    return {
        'graph:n_vertices': g.number_of_nodes(),
        'graph:n_edges': g.number_of_edges(),
        'graph:max_degree': degrees.max(),
        'graph:min_degree': degrees.min(),
        'graph:mean_degree': degrees.mean(),
        'graph:density': nx.density(g),
        'graph:diameter': nx.diameter(g),
        'graph:is_tree': nx.is_tree(g),
        'graph:is_connected': nx.is_connected(g),
    }


def confusion_matrix(
        ground: Iterable[Any], condition_positive: Container[Any], prediction_positive: Container[Any]
) -> Tuple[int, int, int, int]:  # TODO: fix the docstrings!
    """
    Compute a confusion matrix for a binary condition.

    Args:
        ground: The ground set of elements to count
        condition_positive: A container holding the condition-positive elements of :literal:`ground`
        prediction_positive: A container holding the prediction-positive elements of ``ground``

    Returns:
        A ``Tuple[int, int, int, int]`` containing the true positive, false negative, false positive, and true negative
        counts, respectively.
    """
    tp, fn, fp, tn = 0, 0, 0, 0
    for x in ground:
        if x in condition_positive:
            if x in prediction_positive:
                tp += 1
            else:
                fn += 1
        else:  # condition negative
            if x in prediction_positive:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn


def recall(tp: int, fn: int, fp: int, tn: int) -> float:
    if tp + fn > 0:
        return tp / (tp + fn)
    else:
        return float('nan')


def precision(tp: int, fn: int, fp: int, tn: int) -> float:
    if tp + fp > 0:
        return tp / (tp + fp)
    else:
        return float('nan')


def accuracy(tp: int, fn: int, fp: int, tn: int) -> float:
    if tp + tn + fp + fn > 0:
        return (tp + tn) / (tp + tn + fp + fn)
    else:
        return float('nan')


def f1score(tp: int, fn: int, fp: int, tn: int) -> float:
    if 2*tp + fp + fn > 0:
        return 2*tp / (2*tp + fp + fn)
    else:
        return float('nan') 


def jaccard_distance(first: Set[Any], second: Set[Any]) -> float:  # TODO: fix the docstrings!
    """
    Compute the Jaccard

    Args:
        first:
        second:

    Returns:
        The Jaccard distance of ``first`` and ``second``.
    """
    return len(first ^ second) / len(first | second)


def read_intset(path: pathlib.Path) -> Set[int]:
    return set(map(int, path.read_text().strip().split(',')))


def write_iterable(path: pathlib.Path, s: Iterable[Any]) -> None:
    path.write_text(','.join(map(str, s)))
