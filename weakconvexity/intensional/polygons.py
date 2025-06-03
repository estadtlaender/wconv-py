from typing import Self

import numpy as np

from scipy.spatial import distance_matrix, ConvexHull

from weakconvexity import ThetaType, Point, Block

EPSILON = 1e-6

class Context:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.distances = distance_matrix(coordinates, coordinates)

    def distance(self, i: int, j: int) -> ThetaType:
        return self.distances[i, j]

    def trace(self, hull: ConvexHull) -> frozenset[int]:
        normal = hull.equations[:, :-1]
        offset = hull.equations[:, -1]
        return frozenset(np.where(np.all(np.dot(self.coordinates, normal.T) + offset <= EPSILON, axis=1))[0])


class Vec(Point):
    def __init__(self, ctx: Context, idx: int):
        self.ctx = ctx
        self.idx = idx

    def distance(self, other: Self) -> ThetaType:
        return self.ctx.distance(self.idx, other.idx)

    def singleton(self) -> 'ConvexPolygon':
        return ConvexPolygon.singleton(self)

    def __eq__(self, other) -> bool:
        if isinstance(other, Vec):
            return (id(self.ctx) == id(other.ctx)) and (self.idx == other.idx)
        else:
            return False

    def __hash__(self) -> int:
        return hash((id(self.ctx), self.idx))


class ConvexPolygon(Block[Vec]):
    def __init__(self, ctx: Context, points: frozenset[int]):
        self.ctx = ctx
        self.generators = tuple(sorted(points))
        self.hull = ConvexHull(ctx.coordinates[self.generators])
        self.indices = self.ctx.trace(self.hull)

    @classmethod
    def singleton(cls, point: Vec) -> Self:
        return ConvexPolygon(point.ctx, frozenset({point.idx}))

    def membership(self, point: Vec) -> bool:
        return point.idx in self.indices

    def distance(self, other: Self) -> ThetaType:
        raise NotImplementedError("todo!")

    def join(self, other: Self) -> Self:
        return ConvexPolygon(self.ctx, self.indices | other.indices)

    def connectivity_index(self) -> ThetaType:
        return EPSILON

    def __eq__(self, other) -> bool:
        if isinstance(other, ConvexPolygon):
            return (id(self.ctx) == id(other.ctx)) and (self.indices == other.indices)
        else:
            return False

    def __hash__(self) -> int:
        return hash((id(self.ctx), self.indices))