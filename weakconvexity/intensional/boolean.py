import itertools as it

from typing import Self, Any
from collections.abc import Iterable, Sequence, Callable

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sortedcontainers import SortedDict, SortedKeysView

from weakconvexity import ThetaType, Point, Block
from weakconvexity.intensional import wconv, chf as core_chf, heuristic_chf as core_heuristic_chf


class BinVec(Point, Sequence[bool]):
    def __init__(self, bits: Iterable[bool]):
        self._bits: tuple[bool, ...] = tuple(bits)

    @classmethod
    def from_any(cls, bits: Iterable[Any]) -> Self:
        return BinVec(bool(i) for i in bits)

    def distance(self, other: Self) -> ThetaType:
        return sum(x != y for x, y in zip(self._bits, other._bits))

    def singleton(self) -> 'Conjunction':
        return Conjunction(idx if val else -idx for idx, val in enumerate(self, start=1))

    def __getitem__(self, index: int) -> bool:
        return self._bits[index]

    def __len__(self) -> int:
        return len(self._bits)

    def __hash__(self) -> int:
        return hash(self._bits)

    def __eq__(self, other) -> bool:
        if isinstance(other, BinVec):
            return self._bits == other._bits
        else:
            return False

    def __str__(self) -> str:
        return 'BinVec[' + ','.join('1' if bit else '0' for bit in self._bits) + ']'

    def __repr__(self) -> str:
        return str(self)


class Conjunction(Block[BinVec], Callable[[BinVec], bool]):
    def __init__(self, literals: Iterable[int]):
        self.literals: frozenset[int] = frozenset(literals)
        for literal in self.literals:
            if -literal in self.literals:
                self.literals = frozenset({0})
                break

    def evaluate(self, assignment: BinVec) -> bool:
        if self.is_contradiction():
            return False
        else:
            for literal in self.literals:
                if (literal > 0) != assignment[abs(literal) - 1]:
                    return False
            return True

    def is_tautology(self) -> bool:
        return not bool(self.literals)

    def is_contradiction(self) -> bool:
        return 0 in self.literals

    def distance(self, other: Self) -> int:
        count = 0
        for i in self.literals:
            if -i in other.literals:
                count += 1
        return count

    @classmethod
    def singleton(cls, p: BinVec) -> Self:
        return Conjunction(idx if val else -idx for idx, val in enumerate(p, start=1))

    def membership(self, p: BinVec) -> bool:
        return self.evaluate(p)

    def join(self, other: Self) -> Self:
        return Conjunction(self.literals & other.literals)

    def connectivity_index(self) -> ThetaType:
        return 2

    def __hash__(self) -> int:
        return hash(self.literals)

    def __call__(self, assignment: BinVec) -> bool:
        return self.evaluate(assignment)

    def __str__(self) -> str:
        if self.is_contradiction():
            return '\u22a5'  # \u22a5: ⊥  (representing "always false")
        elif not self.literals:
            return '\u22a4'  # \u22a4: ⊤  (representing "always true")
        else:
            return " \u2227 ".join(  # \u2227: ∧
                f"v[{lit}]" if lit > 0 else f"\u00acv[{-lit}]"  # \u00ac: ¬  (representing "not")
                for lit in sorted(self.literals, key=lambda x: (abs(x), x))
            )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, Conjunction):
            return self.literals == other.literals
        else:
            return False

    def __len__(self) -> int:
        if 0 in self.literals:
            raise ValueError("Contradiction has no length")
        else:
            return len(self.literals)

    def generate_positive_examples(
            self, n_vars: int, n_examples: int = 1, rng: None | int | np.random.Generator = None
    ) -> BinVec | frozenset[BinVec]:
        if rng is None or isinstance(rng, int):
            rng = np.random.default_rng(seed=rng)

        positives: set[BinVec] = set()
        while (n_new := n_examples - len(positives)) > 0:
            bits = rng.choice([0, 1], size=(n_new, n_vars), replace=True)
            for lit in self.literals:
                bits[:, abs(lit) - 1] = 0 if lit < 0 else 1
            positives.update(BinVec.from_any(vector) for vector in bits)

        return frozenset(positives) if n_examples > 1 else positives.pop()


def evaluate(dnf: frozenset[Conjunction], x: BinVec) -> bool:
    """
    Evaluate a given DNF at a given point.
    Args:
        dnf: the DNF to evaluate
        x: the point

    Returns:
        ``True`` if the respective DNF evaluates to ``True`` at the given point, ``False`` otherwise.
    """
    return any(term.evaluate(x) for term in dnf)


def random_literals(
        n_vars: int, expected_length: int, p_positive: float = 0.5, rng: int | np.random.Generator | None = None
) -> set[int]:
    """
    Generate a random set of literals with an expected length.
    Args:
        n_vars: the number of variables to choose from
        expected_length: the expected number of positive or negative literals occurring in the result
        p_positive: the probability that a literal occurs positively in the result (default: ``0.5``)
        rng: If ``int``, then used as a seed for ``np.random.Generator``. If ``np.random.Generator``, then used as is.
             If ``None``, then a seedless ``np.random.Generator`` is initialized.

    Returns:
        A ``list[int]`` containing the generated literal values. E.g., -4 represents the literal ¬v[4]
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    p_literal = expected_length / n_vars
    literals = np.where(rng.uniform(size=n_vars) < p_literal)[0] + 1
    signs = rng.choice([-1, 1], size=len(literals), replace=True, p=[1 - p_positive, p_positive])

    return set(signs * literals)


def random_conjunction(
        n_vars: int, expected_length: int, p_positive: float = 0.5, rng: int | np.random.Generator | None = None
) -> Conjunction:
    """
    Generate a random conjunction over a number of literals of an expected length.
    Args:
        n_vars: number of variables in the Hamming space
        expected_length: the expected number of literals in the resulting ``Conjunction``
        p_positive: the probability that a literal is positive
        rng: If ``int``, then used as a seed for a ``np.random.Generator``. If ``np.random.Generator``, then used as is.
             If ``None``, then a seedless ``np.random.Generator`` is initialized.

    Returns:
        A random ``Conjunction`` over ``n_vars`` variables, on average containing ``expected_length`` literals of which
        roughly ``p_positive`` are positive.
    """
    return Conjunction(random_literals(n_vars, expected_length, p_positive, rng))


def random_distant_dnf(
        n_vars: int, n_terms: int, expected_term_length: int, theta_min: ThetaType,
        theta_max: ThetaType, rng: int | np.random.Generator | None = None
) -> tuple[frozenset[Conjunction], int]:
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    # generate dnf literals
    terms = [random_literals(n_vars, expected_term_length, p_positive=0.5, rng=rng) for _ in range(n_terms)]

    # enforce distance condition for theta
    n_additions = 0
    for first, second in it.combinations(terms, r=2):
        local_theta = rng.integers(theta_min, theta_max, endpoint=True)
        distance = sum(-lit in second for lit in first)

        if distance < local_theta:
            # need to introduce more conflicts
            n_new_additions = int(local_theta - distance)
            low = n_vars + n_additions + 1
            high = low + n_new_additions
            signs = rng.choice([-1, 1], size=n_new_additions, replace=True)
            new_literals = signs * np.arange(low, high)
            first.update(new_literals)
            second.update(-new_literals)
            n_additions += n_new_additions

    return frozenset(map(lambda literals: Conjunction(literals), terms)), n_vars + n_additions


def random_dnf(
        n_vars: int, n_terms: int, expected_term_length: int, p_positive: float = 0.5,
        rng: int | np.random.Generator | None = None
) -> frozenset[Conjunction]:
    """
    Generate a random DNF with a given number of terms of a given expected length over a given number of variables.
    Args:
        n_vars: the dimension of the underlying Hamming space
        n_terms: the number of terms in the DNF
        expected_term_length: the expected length of each term in the DNF
        p_positive: the probability that a literal in a term of the DNF is positive
        rng: If ``int``, then used as a seed for a ``np.random.Generator``. If ``np.random.Generator``, then used as is.
             If ``None``, then a seedless ``np.random.Generator`` is initialized.

    Returns:
        A ``frozenset[Conjunction]`` representing the generated DNF.
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    terms = [random_literals(n_vars, expected_term_length, p_positive=p_positive, rng=rng) for _ in range(n_terms)]

    return frozenset(map(lambda literals: Conjunction(literals), terms))


# def wconv(theta: ThetaType, generators: Iterable[BinVec]) -> frozenset[Conjunction]:
#     """
#     Compute the weakly convex hull of the ``generators`` with respect to the convexity strength ``theta``.
#     Args:
#         theta: convexity strength which acts as a global distance threshold for the convexity condition
#         generators: the points of which to compute the weakly convex hull of.
#
#     Returns:
#         A ``frozenset[Conjunction]`` representing a disjunctive normal form (DNF). Each term represents a block.
#         Pairwise, the blocks have a distance of at least ``theta``.
#     """
#     blocks: dict[Conjunction, set[Conjunction]] = {Conjunction.singleton(generator): set() for generator in generators}
#     near_blocks: set[frozenset[Conjunction]] = set()
#     for first, second in it.combinations(blocks.keys(), r=2):
#         if first.distance(second) <= theta:
#             near_blocks.add(frozenset((first, second)))
#             blocks[first].add(second)
#             blocks[second].add(first)
#
#     while near_blocks:
#         # choose blocks to be joined
#         first, second = near_blocks.pop()
#
#         # delete old blocks from data structure
#         for other in blocks[first]:
#             near_blocks.discard(frozenset((other, first)))
#             blocks[other].discard(first)
#         del blocks[first]
#
#         for other in blocks[second]:
#             near_blocks.discard(frozenset((other, second)))
#             blocks[other].discard(second)
#         del blocks[second]
#
#         # compute new block and add to data structure
#         new = first.join(second)
#         blocks[new] = set()
#
#         # add connections to data structure
#         for other, adjacent in blocks.items():
#             if (other != new) and (new.distance(other) <= theta):
#                 near_blocks.add(frozenset((new, other)))
#                 blocks[other].add(new)
#                 blocks[new].add(other)
#
#     return frozenset(blocks.keys())
#
#
# def chf(positives: Iterable[BinVec], negatives: Iterable[BinVec], tau: ThetaType = 2) -> None | frozenset[Conjunction]:
#     """
#     Find a weakly convex Boolean function that is consistent with the given ``positives`` and ``negatives``.
#     Args:
#         positives: positives examples (i.e., members of the unknown target concept)
#         negatives: negative examples (i.e., members of the unknown target concept's complement)
#         tau: minimum global distance threshold between the conjunctions in the result
#
#     Returns:
#         ``None`` if no consistent hypothesis exists. A ``frozenset[Conjunction]`` representing the hypothesis DNF.
#
#     Notes:
#         A weakly convex Boolean function can be represented by a DNF where each term (i.e., conjunction) in the DNF
#         represents one weakly convex block. The pairwise distances between the blocks must be at least ``tau``.
#     """
#     iteration: int = 0
#     blocks: set[Conjunction] = {Conjunction.singleton(p) for p in positives}
#     pairs: SortedDict[ThetaType, list[frozenset[Conjunction]]] = SortedDict()
#     distances: SortedKeysView = pairs.keys()
#     already_joined: set[int] = set()
#
#     for first, second in it.combinations(blocks, r=2):
#         dist = first.distance(second)
#         if dist in pairs:
#             pairs[dist].append(frozenset((first, second)))
#         else:
#             pairs[dist] = [frozenset((first, second))]
#
#     first_min_dist = None
#     while len(blocks) >= 2:
#         if first_min_dist is None:
#             first_min_dist = distances[0]
#         theta = max(tau, distances[0])  # distances[0] is the minimum element in the heap
#
#         old_blocks: set[Conjunction] = set(blocks)
#         while distances and ((min_dist := distances[0]) <= theta):
#             first, second = pairs[min_dist].pop()
#             if not pairs[min_dist]:
#                 del pairs[min_dist]
#
#             if id(first) in already_joined or id(second) in already_joined:
#                 continue
#
#             # compute join for selected blocks
#             new_block = first.join(second)
#             if any(new_block.membership(e) for e in negatives):
#                 # inconsistent
#                 if iteration == 0:
#                     if first_min_dist <= tau:
#                         return None
#                     else:
#                         return frozenset(old_blocks)
#                 else:
#                     return frozenset(old_blocks)
#
#             # update data structures
#             already_joined.add(id(first))
#             already_joined.add(id(second))
#             blocks.remove(first)
#             blocks.remove(second)
#             for other in blocks:
#                 dist = new_block.distance(other)
#                 if dist in pairs:
#                     pairs[dist].append(frozenset((new_block, other)))
#                 else:
#                     pairs[dist] = [frozenset((new_block, other))]
#             blocks.add(new_block)
#
#         iteration += 1
#
#     return frozenset(blocks)

def chf(positives: Iterable[BinVec], negatives: Iterable[BinVec], tau: ThetaType = 2) -> None | frozenset[Conjunction]:
    return core_chf(positives, negatives, tau)


# def heuristic_chf(
#         positives: Iterable[BinVec], negatives: Iterable[BinVec], tau: ThetaType = 2
# ) -> None | frozenset[tuple[Conjunction, ThetaType]]:
#     """
#     Use a heuristic to find an at least ``tau`` variational weakly convex Boolean function that is consistent with the
#     given ``positives`` and ``negatives``.
#     Args:
#         positives: positive examples drawn from an unknown target concept
#         negatives: negative examples drawn from an unknown target concept
#         tau: the minimum local distance threshold of the weakly convex blocks
#
#     Returns:
#         A ``frozenset[tuple[Conjunction, ThetaType]]`` where each element in the set is a term of the DNF together with
#         its local distance threshold.
#
#     Notes:
#          Notice that for the Hamming space, ``tau`` is equal to ``2`` everywhere. The implementation, however, supports
#          setting ``tau`` manually to some other value in case the application scenario requires *larger* values than
#          ``2``. You **must not** use smaller values!
#     """
#     blocks: set[Conjunction] = set(wconv(tau, positives))
#
#     # check consistency of the tau-convex hull
#     # we don't stand a chance if this is already inconsistent with the negative examples
#     for block in blocks:
#         for e in negatives:
#             if block.membership(e):
#                 return None
#
#     theta: dict[Conjunction, ThetaType] = {block: tau for block in blocks}
#
#     # initialize data structures
#     pairs: SortedDict[ThetaType, list[frozenset[Conjunction]]] = SortedDict()
#     distances: SortedKeysView = pairs.keys()  # remark: this view is updated, reflecting changes
#     for first, second in it.combinations(blocks, r=2):
#         dist = first.distance(second)
#         if dist in pairs:
#             pairs[dist].append(frozenset((first, second)))
#         else:
#             pairs[dist] = [frozenset((first, second))]
#
#     already_joined: set[Conjunction] = set()
#     forbidden_joins: set[frozenset[Conjunction]] = set()
#
#     while pairs:
#         min_dist = distances[0]
#         pair = pairs[min_dist].pop()
#         if not pairs[min_dist]:
#             del pairs[min_dist]
#
#         if (not pair.isdisjoint(already_joined)) or (pair in forbidden_joins):
#             continue
#
#         first, second = pair
#
#         # compute _full_ join, i.e., all consequential joins that are required after the first one
#         new_block = first.join(second)
#         theta_new_block = new_block.connectivity_index()
#         new_joins: set[Conjunction] = {first, second}
#
#         while (required := {block for block in blocks
#                             if (block not in new_joins)
#                             and (new_block.distance(block) <= max(tau, theta_new_block, theta[block]))}):
#             for other in required:
#                 new_block = new_block.join(other)
#                 theta_new_block = new_block.connectivity_index()
#                 new_joins.add(other)
#
#         # check with negative examples, were we allowed to join first and second in the first place?
#         if any(new_block.membership(e) for e in negatives):
#             # inconsistent -> disallow to join `first` and `second`
#             forbidden_joins.add(frozenset((first, second)))
#         else:
#             # consistent -> make transaction
#             already_joined.update(new_joins)
#             blocks.difference_update(new_joins)
#             for other in new_joins:
#                 del theta[other]
#
#             for other in blocks:
#                 dist = new_block.distance(other)
#                 if dist in pairs:
#                     pairs[dist].append(frozenset((new_block, other)))
#                 else:
#                     pairs[dist] = [frozenset((new_block, other))]
#
#             blocks.add(new_block)
#             theta[new_block] = theta_new_block
#
#     return frozenset((block, theta[block]) for block in blocks)


def heuristic_chf(
        positives: Iterable[BinVec], negatives: Iterable[BinVec], tau: ThetaType = 2
) -> None | frozenset[tuple[Conjunction, ThetaType]]:
    return core_heuristic_chf(positives, negatives, tau)


def confusion_matrix(
        n_vars: int, target: frozenset[Conjunction], hypothesis: frozenset[Conjunction]
) -> tuple[int, int, int, int]:
    """
    Compute the confusion matrix for a given (unknown) ``target`` and a given ``hypothesis`` for a given
    number ``n_vars`` of variables.
    Args:
        n_vars: the number of variables
        target: the unknown target concept determining the actual condition
        hypothesis: the hypothesis to test

    Returns:
        A ``tuple[int, int, int, int]`` with the respective confusion matrix counts.

    Notes:
        - This function assumes that the terms of both given DNFs are pairwise disjoint. This is the case for
          variational weakly convex Boolean functions.
        - The order of the returned tuple follows the same order as in the ``sklearn`` package.
    """
    condition_pos = 0
    for term in target:
        condition_pos += 2**(n_vars - len(term))

    prediction_pos = 0
    for term in hypothesis:
        prediction_pos += 2**(n_vars - len(term))

    tp = 0
    for a, b in it.product(target, hypothesis):
        if a.distance(b) == 0:
            tp += 2**(n_vars - len(Conjunction(a.literals | b.literals)))

    fn = condition_pos - tp
    fp = prediction_pos - tp

    tn = 2**n_vars - (tp + fp + fn)

    return tn, fp, fn, tp


def jaccard_distance(n_vars: int, first: frozenset[Conjunction], second: frozenset[Conjunction]) -> float:
    """
    Compute the Jaccard distance between two DNFs under the assumption that the terms in the DNFs are pairwise disjoint.
    Args:
        n_vars: the dimension of the Hamming space
        first: the first DNF
        second: the second DNF

    Returns:
        A ``float`` with the Jaccard distance, i.e., ``1 - len(intersection) / len(union)``
    """
    intersection = 0
    for a, b in it.product(first, second):
        if a.distance(b) == 0:
            intersection += 2**(n_vars - len(Conjunction(a.literals | b.literals)))

    union = 0
    for term in it.chain(first, second):
        union += 2**(n_vars - len(term))
    union -= intersection

    return 1 - (intersection / union)


def generate_positive_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_examples: int, weighted_terms: bool = True,
        rng: None | int | np.random.Generator = None
) -> frozenset[BinVec]:
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    dnf_terms = tuple(dnf)
    weights = np.array([2**(n_vars - len(term)) for term in dnf_terms], dtype=float)
    weights /= np.sum(weights)  # normalize so that the weights sum up to 1

    positives = set()
    while (n_missing := (n_examples - len(positives))) > 0:
        choice = rng.choice(dnf_terms, size=n_missing, replace=True, p=weights if weighted_terms else None)
        positives.update(term.generate_positive_examples(n_vars, rng=rng) for term in choice)

    return frozenset(positives)


def generate_negative_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_examples: int, inject_beneficial_per_pair: int = 0,
        rng: None | int | np.random.Generator = None
) -> frozenset[BinVec]:
    """
    Generate negative examples for a DNF concept.
    Args:
        dnf: the DNF to generate the examples for
        n_vars: the dimension of the ground space
        n_examples: the number of examples to generate
        inject_beneficial_per_pair: the number of beneficial examples to inject for each pair of terms in the DNF
        rng: the PRNG to use for example generation

    Returns:
        A ``frozenset[BinVec]`` representing the immutable set of negative examples that were chosen.

    Notes:
         The beneficial examples are chosen from the join of two terms as long as they are negative for _all_ terms in
         the DNF. All other examples are then chosen uniformly. Due to the imbalanced nature of the weakly convex DNFs
         we are considering, one can generate negative examples uniformly by generating bit vectors uniformly and
         checking whether they satisfy the DNF.

         Notice that the pairs from which beneficial examples are drawn are sorted by distance in ascending order. In
         other words, near pairs are favored.
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    negatives = set()

    # beneficial negative examples
    n_pairs = len(dnf) * (len(dnf) - 1) // 2  # formula for "len(dnf) choose 2"
    n_pairs_to_consider = 0 if inject_beneficial_per_pair == 0 else (
        n_pairs if n_pairs * inject_beneficial_per_pair <= n_examples else
        int(np.ceil(n_examples / inject_beneficial_per_pair))
    )
    for a, b in it.islice(
            sorted(it.combinations(dnf, r=2), key=lambda pair: pair[0].distance(pair[1])),
            n_pairs_to_consider
    ):
        n_negatives = len(negatives)
        join = a.join(b)
        while (n_missing := (min(n_negatives + inject_beneficial_per_pair, n_examples) - len(negatives))) > 0:
            examples = join.generate_positive_examples(n_vars, n_missing, rng=rng)
            if n_missing == 1:
                examples = frozenset({examples})
            negatives.update(example for example in examples if not any(term.evaluate(example) for term in dnf))

    # uniform negative examples
    while (n_missing := (n_examples - len(negatives))) > 0:
        bits = rng.choice([0, 1], size=(n_missing, n_vars), replace=True)
        negatives.update(
            bv for vector in bits if not any(term.evaluate(bv := BinVec.from_any(vector)) for term in dnf)
        )

    return frozenset(negatives)


def generate_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_examples: int | tuple[int, int],
        weighted_terms: bool = True, inject_beneficial_per_pair: int = 0,
        rng: None | int | np.random.Generator = None
) -> tuple[frozenset[BinVec], frozenset[BinVec]]:
    """
    Generate positive and negative examples for a given DNF concept.
    Args:
        dnf: the DNF to generate examples for
        n_vars: the dimension of the ground space
        n_examples: if ``int`` the total number of examples to generate. In this case the positive and negative
                    examples are roughly balanced. if ``tuple[int, int]``, the number of positive and negative examples
                    to generate, respectively.
        weighted_terms: flag to indicate whether each term should be weighted with its volume
        inject_beneficial_per_pair:
        rng:

    Returns:
        A ``tuple[frozenset[BinVec], frozenset[BinVec]]`` representing the positive and negative examples,
        respectively.
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    if isinstance(n_examples, int):
        n_positive_examples = n_examples // 2
        n_negative_examples = n_examples - n_positive_examples
    elif isinstance(n_examples, tuple):
        n_positive_examples = n_examples[0]
        n_negative_examples = n_examples[1]
    else:
        raise ValueError("invalid type for n_examples given")

    positives = generate_positive_examples(dnf, n_vars, n_positive_examples, weighted_terms, rng=rng)
    negatives = generate_negative_examples(dnf, n_vars, n_negative_examples, inject_beneficial_per_pair, rng=rng)

    return positives, negatives


def generate_uniform_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_examples: int | tuple[int, int],
        rng: None | int | np.random.Generator = None
) -> tuple[frozenset[BinVec], frozenset[BinVec]]:
    return generate_examples(dnf, n_vars, n_examples, weighted_terms=True, inject_beneficial_per_pair=0, rng=rng)


def generate_beneficial_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_pos_examples_per_term: int, n_neg_examples_per_pair: int,
        fill_uniform: bool = False, rng: None | int | np.random.Generator = None
) -> tuple[frozenset[BinVec], frozenset[BinVec]]:
    """
    Generate beneficial positive and negative examples for a given DNF.
    Args:
        dnf: the given concept to generate the examples for
        n_vars: the number of literals in the ground space.
        n_pos_examples_per_term: the number of positive examples per term to generate
        n_neg_examples_per_pair: the number of negative examples per pair of terms to generate
        fill_uniform: If ``True``, the negative examples are filled with uniformly chosen negative examples to match
                      the number of positive examples. If ``False`` no additional negative examples are added.
        rng: the ``np.random.Generator`` to use for randomness. If ``int``, this number is used to seed a default
             generator. If ``None``, the default generator is initialized

    Returns:
        A ``tuple[frozenset[BinVec], frozenset[BinVec]]``. The first part contains the positive, the second part the
        negative examples.

    Notes:
        This method may enter an infinite loop if the dimension of any term in the DNF is too small, i.e., the number
        of true points is less than ``n_pos_examples_per_term``. The same holds true when ``n_neg_examples_per_pair``
        is chosen too large.

        Depending on the parameters, we might end up with more negative examples
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(seed=rng)

    positives = set()
    for term in dnf:
        positives.update(
            term.generate_positive_examples(n_vars, n_pos_examples_per_term, rng)
        )

    negatives = set()
    for a, b in it.combinations(dnf, r=2):
        n_negatives = len(negatives)
        join = a.join(b)
        while len(negatives) < n_negatives + n_neg_examples_per_pair:
            n_missing = n_neg_examples_per_pair - (len(negatives) - n_negatives)
            examples = join.generate_positive_examples(n_vars, n_missing, rng)
            if n_missing == 1:
                examples = frozenset({examples})
            negatives.update(example for example in examples if not any(term.evaluate(example) for term in dnf))

    if fill_uniform:
        while (n_missing := len(positives) - len(negatives)) > 0:
            bitvectors = rng.choice([0, 1], size=(n_missing, n_vars), replace=True)
            examples = map(BinVec.from_any, bitvectors)
            negatives.update(example for example in examples if not any(term.evaluate(example) for term in dnf))

    return frozenset(positives), frozenset(negatives)


def generate_near_boundary_examples(
        dnf: frozenset[Conjunction], n_vars: int, n_examples: int | tuple[int, int], max_dist: int = 1,
        rng: None | int | np.random.Generator = None
) -> tuple[frozenset[BinVec], frozenset[BinVec]]:
    if rng is None or isinstance(rng, int):
        rng =np.random.default_rng(seed=rng)

    if isinstance(n_examples, int):
        n_positives = n_examples
        n_negatives = n_examples
    elif isinstance(n_examples, tuple):
        n_positives, n_negatives = n_examples
    else:
        raise ValueError("invalid type for n_examples given")

    positives = generate_positive_examples(dnf, n_vars, n_positives, weighted_terms=True, rng=rng)

    intermediates = generate_positive_examples(dnf, n_vars, n_negatives, weighted_terms=True, rng=rng)
    negatives = set()
    for intermediate in intermediates:
        negative = BinVec(intermediate)  # copy
        dist = rng.integers(1, max_dist, endpoint=True)
        while evaluate(dnf, negative):
            flips = rng.choice(len(negative), size=dist, replace=False)
            negative = BinVec([not intermediate[i] if i in flips else intermediate[i] for i in range(len(intermediate))])
        negatives.add(negative)

    return frozenset(positives), frozenset(negatives)


def examples_to_numpy(
        positives: frozenset[BinVec], negatives: frozenset[BinVec], shuffled: bool = False,
        rng: None | np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts two ``frozenset`` of positive and negative examples into two numpy arrays (``np.ndarray``).
    Args:
        positives:
            set of positive examples
        negatives:
            set of negative examples
        shuffled:
            Boolean flag whether or not to shuffle the numpy array before returning it
        rng:
            PRNG to be used for shuffling. If ``None``, a new PRNG is initialized with a random seed.

    Returns:
        A Boolean numpy array with
    """
    data = np.array(
        [list(v) + [True] for v in positives] +
        [list(v) + [False] for v in negatives]
    )

    if shuffled and rng is not None:
        rng.shuffle(data)

    X = data[:, :-1]
    y = data[:, -1]

    return X, y


def convert_dtc_to_dnf(dtc: DecisionTreeClassifier) -> frozenset[Conjunction]:
    """
    Converts the given ``DecisionTreeClassifier`` into a DNF representation.

    Internally, it finds all paths to positively labelled leaves.

    Args:
        dtc: The ``DecisionTreeClassifier`` to be converted.

    Returns:
        A ``frozenset[wc.Conjunction]`` which is a set of terms/conjunctions. Together they represent the DNF.
    """
    tree = dtc.tree_

    def _is_leaf(node: int) -> bool:
        return tree.children_left[node] == tree.children_right[node]

    def _get_paths(node: int) -> list[tuple]:
        if _is_leaf(node):
            return [((node, 'leaf'),)]
        else:
            paths = list()
            if tree.children_left[node] != -1:
                paths.extend(((node, 'left'), *path) for path in _get_paths(tree.children_left[node]))

            if tree.children_right[node] != -1:
                paths.extend(((node, 'right'), *path) for path in _get_paths(tree.children_right[node]))

            return paths

    def _get_path_label(path):
        leaf, _ = path[-1]
        return tree.value[leaf][0].argmax()

    def _to_conjunction(path):
        literals = []
        for (i, dir) in path:
            match dir:
                case 'right':
                    literals.append((tree.feature[i] + 1))
                case 'left':
                    literals.append(-(tree.feature[i] + 1))
                case 'leaf':
                    pass
        return Conjunction(literals)

    return frozenset(
        map(
            _to_conjunction,
            filter(lambda path: _get_path_label(path) == 1, _get_paths(0))
        )  # 0 represents the root node
    )
