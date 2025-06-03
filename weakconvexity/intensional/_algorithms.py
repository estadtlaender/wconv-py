import itertools as it
from typing import Iterable
from dataclasses import dataclass

import numpy as np
from sortedcontainers import SortedDict, SortedKeysView

from weakconvexity import ThetaType, P, Block


def wconv(theta: ThetaType, generators: Iterable[P]) -> frozenset[Block[P]]:
    blocks: dict[Block[P], set[Block[P]]] = {generator.singleton(): set() for generator in generators}
    near_blocks: set[frozenset[Block[P]]] = set()
    for first, second in it.combinations(blocks.keys(), r=2):
        if first.distance(second) <= theta:
            near_blocks.add(frozenset((first, second)))
            blocks[first].add(second)
            blocks[second].add(first)

    while near_blocks:
        # choose blocks to be joined
        first, second = near_blocks.pop()

        # delete old blocks from data structure
        for other in blocks[first]:
            near_blocks.discard(frozenset((other, first)))
            blocks[other].discard(first)
        del blocks[first]

        for other in blocks[second]:
            near_blocks.discard(frozenset((other, second)))
            blocks[other].discard(second)
        del blocks[second]

        # compute new block and add to data structure
        new = first.join(second)
        blocks[new] = set()

        # add connections to data structure
        for other, adjacent in blocks.items():
            if (other != new) and (new.distance(other) <= theta):
                near_blocks.add(frozenset((new, other)))
                blocks[other].add(new)
                blocks[new].add(other)

    return frozenset(blocks.keys())


def chf(positives: Iterable[P], negatives: Iterable[P], tau: ThetaType) -> None | frozenset[Block[P]]:
    iteration: int = 0
    blocks: set[Block[P]] = {p.singleton() for p in positives}
    pairs: SortedDict[ThetaType, list[frozenset[Block[P]]]] = SortedDict()
    distances: SortedKeysView = pairs.keys()
    already_joined: set[int] = set()

    for first, second in it.combinations(blocks, r=2):
        dist = first.distance(second)
        if dist in pairs:
            pairs[dist].append(frozenset((first, second)))
        else:
            pairs[dist] = [frozenset((first, second))]

    first_min_dist = None
    while len(blocks) >= 2:
        if first_min_dist is None:
            first_min_dist = distances[0]
        theta = max(tau, distances[0])  # distances[0] is the minimum element in the heap

        old_blocks: set[Block[P]] = set(blocks)
        while distances and ((min_dist := distances[0]) <= theta):
            first, second = pairs[min_dist].pop()
            if not pairs[min_dist]:
                del pairs[min_dist]

            if id(first) in already_joined or id(second) in already_joined:
                continue

            # compute join for selected blocks
            new_block = first.join(second)
            if any(new_block.membership(e) for e in negatives):
                # inconsistent
                if iteration == 0:
                    if first_min_dist <= tau:
                        return None
                    else:
                        return frozenset(old_blocks)
                else:
                    return frozenset(old_blocks)

            # update data structures
            already_joined.add(id(first))
            already_joined.add(id(second))
            blocks.remove(first)
            blocks.remove(second)
            for other in blocks:
                dist = new_block.distance(other)
                if dist in pairs:
                    pairs[dist].append(frozenset((new_block, other)))
                else:
                    pairs[dist] = [frozenset((new_block, other))]
            blocks.add(new_block)

        iteration += 1

    return frozenset(blocks)


def heuristic_chf(
        positives: Iterable[P], negatives: Iterable[P], tau: ThetaType
) -> None | frozenset[tuple[Block[P], ThetaType]]:
    blocks: set[Block[P]] = set(wconv(tau, positives))

    # check consistency of the tau-convex hull
    # we don't stand a chance if this is already inconsistent with the negative examples
    for block in blocks:
        for e in negatives:
            if block.membership(e):
                return None

    theta: dict[Block[P], ThetaType] = {block: tau for block in blocks}

    # initialize data structures
    pairs: SortedDict[ThetaType, list[frozenset[Block[P]]]] = SortedDict()
    distances: SortedKeysView = pairs.keys()  # remark: this view is updated, reflecting changes
    for first, second in it.combinations(blocks, r=2):
        dist = first.distance(second)
        if dist in pairs:
            pairs[dist].append(frozenset((first, second)))
        else:
            pairs[dist] = [frozenset((first, second))]

    already_joined: set[Block[P]] = set()
    forbidden_joins: set[frozenset[Block[P]]] = set()

    while pairs:
        min_dist = distances[0]
        pair = pairs[min_dist].pop()
        if not pairs[min_dist]:
            del pairs[min_dist]

        if (not pair.isdisjoint(already_joined)) or (pair in forbidden_joins):
            continue

        first, second = pair

        # compute _full_ join, i.e., all consequential joins that are required after the first one
        new_block = first.join(second)
        theta_new_block = new_block.connectivity_index()
        new_joins: set[Block[P]] = {first, second}

        while (required := {block for block in blocks
                            if (block not in new_joins)
                               and (new_block.distance(block) <= max(tau, theta_new_block, theta[block]))}):
            for other in required:
                new_block = new_block.join(other)
                theta_new_block = new_block.connectivity_index()
                new_joins.add(other)

        # check with negative examples, were we allowed to join first and second in the first place?
        if any(new_block.membership(e) for e in negatives):
            # inconsistent -> disallow to join `first` and `second`
            forbidden_joins.add(frozenset((first, second)))
        else:
            # consistent -> make transaction
            already_joined.update(new_joins)
            blocks.difference_update(new_joins)
            for other in new_joins:
                del theta[other]

            for other in blocks:
                dist = new_block.distance(other)
                if dist in pairs:
                    pairs[dist].append(frozenset((new_block, other)))
                else:
                    pairs[dist] = [frozenset((new_block, other))]

            blocks.add(new_block)
            theta[new_block] = theta_new_block

    return frozenset((block, theta[block]) for block in blocks)


@dataclass
class _Info:
    label: int
    theta: ThetaType


def classification(
        X: Iterable[P], y: Iterable[int], tau: ThetaType
) -> None | frozenset[tuple[Block[P], ThetaType, int]]:
    """
    Multi-label classification of points based on greedy, iterative merging of weakly convex blocks.

    The output is a "monochromatic" locally constrained block system.

    @param X: The input points
    @param y: The labels of the input points, same order as X
    @param tau: The global minimum distance parameter which is domain-specific
    @return: A locally constrained block system whose blocks are homogeneous with respect to the contained labels
    """
    return classification_from_blocks(inblocks=map(lambda pl: (pl[0].singleton(), tau, pl[1]), zip(X, y)), tau=tau)


def classification_from_blocks(
        inblocks: Iterable[tuple[Block[P], ThetaType, int]], tau: ThetaType 
) -> None | frozenset[tuple[Block[P], ThetaType, int]]: 
    blocks: dict[Block[P], _Info] = {block: _Info(label=label, theta=theta) for block, theta, label in inblocks}

    pairs: SortedDict[float, list[frozenset[Block[P]]]] = SortedDict()
    distances: SortedKeysView = pairs.keys()
    for first, second in it.combinations(blocks, r=2):
        dist = first.distance(second)
        pairs.setdefault(dist, []).append(frozenset({first, second}))

    joined: set[Block[P]] = set()
    forbidden: set[frozenset[Block[P]]] = set()
    while pairs:
        min_dist = distances[0]
        pair = pairs[min_dist].pop()
        if not pairs[min_dist]:
            del pairs[min_dist]

        first, second = pair
        if (not pair.isdisjoint(joined)) or (pair in forbidden):
            # pair is not up-to-date or forbidden to be joined
            continue
        elif blocks[first].label != blocks[second].label:
            # pair is not monochromatic
            continue

        new_block = first.join(second)
        new_block_info = _Info(label=blocks[first].label, theta=max(tau, new_block.connectivity_index()))
        deleted: set[Block[P]] = {first, second}

        consistent = True 
        while (required := {block for block in blocks
                            if (block not in deleted)
                               and (new_block.distance(block) <= max(tau, new_block_info.theta, blocks[block].theta))}):
            for other in required:
                if blocks[other].label != new_block_info.label:
                    # distance constraint requires non-monochrome merge -> inconsistency
                    consistent = False
                    break
                new_block = new_block.join(other)
                new_block_info.theta = max(tau, new_block.connectivity_index())
                deleted.add(other)

            if not consistent:
                break

        if not consistent:
            forbidden.add(pair)
        else:
            # so far, we are consistent -> make transaction
            joined.update(deleted)
            for other in deleted:
                del blocks[other]

            for other in blocks:
                dist = new_block.distance(other)
                pairs.setdefault(dist, []).append(frozenset({new_block, other}))

            blocks[new_block] = new_block_info

    return frozenset({(b, info.theta, info.label) for b, info in blocks.items()})


def classification_randomized(
        X: Iterable[P], y: Iterable[int], tau: float = 1e-10,
        rng: np.random.Generator | int | None = None
) -> None | frozenset[tuple[Block[P], float, int]]:
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    blocks: dict[Block[P], _Info] = {p.singleton(): _Info(label=l, theta=tau) for p, l in zip(X, y)}
    pairs: list[tuple[Block[P], Block[P]]] = list(
        filter(lambda pair: blocks[pair[0]].label == blocks[pair[1]].label, it.combinations(blocks, r=2))
    )
    rng.shuffle(pairs)

    joined: set[Block[P]] = set()
    forbidden: set[frozenset[Block[P]]] = set()
    while pairs:
        pair = pairs.pop()
        first, second = pair
        if first in joined or second in joined or frozenset(pair) in forbidden:
            # the pair is out-of-date or forbidden to be joined -> skip
            continue

        new_block = first.join(second)
        new_block_info = _Info(label=blocks[first].label, theta=max(tau, new_block.connectivity_index()))
        deleted: set[Block[P]] = {first, second}

        consistent = True
        while (required := {block for block, info in blocks.items()
                            if (block not in deleted)
                               and (new_block.distance(block) <= max(tau, new_block_info.theta, blocks[block].theta))}):
            for other in required:
                if blocks[other].label != new_block_info.label:
                    # distance constraint requires non-monochrome merge -> inconsistency
                    consistent = False
                    break
                new_block = new_block.join(other)
                new_block_info.theta = max(tau, new_block.connectivity_index())
                deleted.add(other)

            if not consistent:
                break

        if not consistent:
            forbidden.add(frozenset(pair))
        else:
            # so far, we are consistent -> make transaction
            joined.update(deleted)
            for other in deleted:
                del blocks[other]

            pairs.extend((new_block, other) for other, info in blocks.items() if info.label == new_block_info.label)
            # TODO: decide on another shuffle here! -> rng.shuffle(pairs)
            blocks[new_block] = new_block_info

    return frozenset({(b, info.theta, info.label) for b, info in blocks.items()})