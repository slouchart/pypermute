__version__ = '0.1'
__author__ = 'Sébastien Louchart:sebastien.louchart@gmail.com'

__all__ = ['Permutation', 'PermutationBuilderError', 'PermutationStructuralError']

import random
from typing import Any, Tuple, Union, Dict, Iterator, Sequence, Optional


Cycle = Sequence[int]
Representation = Optional[Sequence[Cycle]]
Map = Dict[int, int]


class PermutationStructuralError(ValueError):
    """Thrown when the representation of a permutation presents some issues
    These issues can be:

    - the same element belongs to more than one cycle
    - one or more elements appear to be missing i.e. the sequence possesses holes
    """
    ...


class PermutationBuilderError(PermissionError):
    """Thrown when `add_cycle` or other mutable methods are called outside of the building context manager
    """
    ...


class Permutation:
    """Provides a way to transpose any ordered set into itself by permuting its elements according to a map
    that is a product of transpositions, cycles and fixed points.

    Allows product of permutations, cycle notation and signature.

    A instance of Permutation is immutable except when entering the building mode
    in which the internal representation is reset and then constructed by repeated calls to the method
    `add_cycle`.
    """

    def __init__(self, representation: Representation = None, reach: int = None):
        """The `representation` of a permutation is a tuple of tuples of integers.

        Each inner tuple is a cycle, each integer must be present once and must be
        in `range(0, reach)` without any hole nor duplicates.

        For instance,

        - `((1,), (0, 2, 3), (4,))` is a valid representation of reach=5

        - `((1, 4), (3,))` is not a valid representation because both 0 and 2 are missing

        - `((0, 2), (3, 4), (1, 2))` is not valid either because 2 is found in two cycles
        """
        self._representation = None
        self._reach = 0

        if representation is not None:
            self._reach = self._validate(representation)
            self._representation = representation
        else:
            assert reach is not None and isinstance(reach, int)
            self._reach = reach

    def add_cycle(self, *args):
        """When in mutable mode, appends a cycle to the permutation being built
        """
        raise PermutationBuilderError('Attempt to add a cycle outside a Permutation build context')

    def invert(self) -> 'Permutation':
        """Returns the inverse permutation such as `p o p-1 = Id`
        """
        with Permutation(reach=self.reach) as ret:
            for cycle in self.canonical():
                ret.add_cycle(*(reversed(cycle)))

        return ret

    @staticmethod
    def compose(a: 'Permutation', b: 'Permutation') -> 'Permutation':
        """Composes two permutations: `a*b(x) <=> a(b(x))`
        """
        assert a.reach == b.reach
        composition_map = dict()
        for elmt in range(0, a.reach):
            composition_map[elmt] = a.transpose(b.transpose(elmt))

        return Permutation.from_map(composition_map)

    @staticmethod
    def from_map(m: Map) -> 'Permutation':
        """Builds a permutation from a dictionary `m`
        """
        reach = len(m)
        if sorted(list(m.keys())) != list(range(0, reach)) or sorted(list(m.values())) != list(range(0, reach)):
            raise ValueError("Given map is not bijective")

        _m = {k: v for k, v in m.items()}  # copy the input map
        with Permutation(reach=reach) as ret:
            while len(_m) > 0:
                current_cycle = []

                # pick an element at random
                first_elem = random.choice(list(_m.keys()))
                next_elem = first_elem
                while True:
                    current_cycle.append(next_elem)
                    next_elem, current_elem = _m[next_elem], next_elem
                    del _m[current_elem]
                    if next_elem == first_elem:
                        ret.add_cycle(*tuple(current_cycle))
                        break

        return ret

    def to_map(self) -> Map:
        """Returns the transposition map of the permutation
        """
        m = dict()
        for inx in range(0, self.reach):
            m[inx] = self.transpose(inx)
        return m

    @staticmethod
    def cyclotomic(vertices: int, allows_fixed_points: bool = True) -> 'Permutation':
        """Returns a permutation on the `vertices`-th roots of one

        if `allows_fixed_points` is True then:

        - the permutation has one fixed point if `vertices` is odd

        - and two fixed points if `vertices` is even

        if `allows_fixed_points` is False then:

        - the permutation is the *complex roots conjugation* if `vertices` is odd,

        - a *circular permutation* if `vertices` is even.
        """
        if vertices <= 0:
            raise ValueError('Number of vertices for a cyclotomic permutation must be 1 or greater')

        representation = []
        if vertices % 2 == 0:
            if allows_fixed_points:
                if vertices > 2:
                    # two fixed points: 0 and vertices // 2
                    representation.append((0,))
                    representation.append((vertices//2,))
                    # vertices - 2 2-cycles
                    for inx in range(1, vertices//2):
                        representation.append((inx, vertices - inx,))
                elif vertices == 2:
                    representation = [(0,), (1,)]
                elif vertices == 1:
                    representation = [(0,)]
            else:
                # circular permutation
                if vertices > 2:
                    cycle = list(range(0, vertices - 1))
                    cycle.insert(0, vertices - 1)
                    representation = [tuple(cycle)]
                elif vertices == 2:
                    representation = [(1, 0)]
                elif vertices == 1:
                    raise ValueError('Cyclotomic permutation of reach 1 with no fixed points is impossible')

        else:
            if allows_fixed_points:
                # one fixed point
                representation = [(0,)]
                if vertices > 2:
                    for inx in range(1, (vertices-1)//2 + 1):
                        representation.append((inx, vertices-inx,))
            else:
                # complex conjugation of non real roots
                if vertices > 1:
                    for inx in range(0, (vertices-1)//2):
                        representation.append((inx, vertices-inx-2,))
                else:
                    raise ValueError('cyclotomic permutation of reach 1 with no fixed points is impossible')

        return Permutation(tuple(representation),)

    def copy(self) -> 'Permutation':
        """Returns a copy of itself
        """
        return Permutation(self._representation)

    def iterated(self, power: int) -> 'Permutation':
        """Composes the permutation with itself `power` times
        """
        p = self.copy()
        while power > 1:
            p *= p
            power -= 1

        return p

    @property
    def reach(self) -> int:
        """Returns the `reach` of a permutation that is the size of the set it acts upon
        """
        return self._reach

    @property
    def order(self) -> int:
        """Returns the `order` of a permutation that is the product of the lengths of its cycles
        """
        p = 1
        for cycle in self._representation:
            p *= cycle.__len__()

        return p

    @property
    def cycles(self) -> int:
        """Returns the total number of cycles of length > 1
        """
        return list(self._iter_cycles(2, None)).__len__()

    def get_cycles(self, order: int = 0) -> Iterator[Cycle]:
        """Returns an iterator of all cycles of order `order`.
        If `order` is 0, returns all the cycles of any order except fixed points
        """
        if order == 0:
            return self._iter_cycles(2, None)
        else:
            return self._iter_cycles(order, order)

    def cycles_by_length(self, length: int) -> int:
        """Returns an iterable of all cycles of a given length
        """
        return list(self.get_cycles(length)).__len__()

    @property
    def transpositions(self) -> int:
        """Returns the number of 2-cycles
        """
        return list(self.get_cycles(2)).__len__()

    def get_transpositions(self) -> Iterator[Cycle]:
        """Returns an iterator on all 2-cycles
        """
        return self.get_cycles(order=2)

    @property
    def fixed_points(self) -> int:
        """Returns the total number of fixed points
        """
        return list(self.get_fixed_points()).__len__()

    def get_fixed_points(self) -> Iterator[Cycle]:
        """returns an iterator on all the fixed points which are just 1-cycles
        """
        return self.get_cycles(order=1)

    def transpose(self, element: int) -> int:
        """Returns the index of the transposed element
        """
        cycle, pos = self._find_element(element)
        return self._find_next(pos, cycle)

    def permute_sequence(self, col: Sequence[Any]) -> Iterator[Any]:
        """Returns a permuted set as an iterator.
        The input parameter must be an **indexable bounded container**
        """
        assert len(col) == self.reach
        index = 0
        for _ in col:
            yield col[self.transpose(index)]
            index += 1

    @staticmethod
    def random_full_cycle(reach: int) -> 'Permutation':
        """Returns a permutation that contains only a `reach`-cycle with random transpositions
        """
        full_cycle = list(range(0, reach))
        random.shuffle(full_cycle)
        representation = tuple(full_cycle)
        with Permutation(reach=len(representation)) as ret:
            ret.add_cycle(*representation)

        return ret

    @staticmethod
    def circular(reach: int, reverse: bool = False) -> 'Permutation':
        """Returns a circular permutation of length `reach`.
        If `reverse` is True, returns the inverse permutation such as
        `circular(n) * circular(n, reverse=True) == identity`
        """
        cycle = list(range(0, reach))
        if reverse:
            cycle = reversed(cycle)
        with Permutation(reach=reach) as p:
            p.add_cycle(*tuple(cycle))

        return p

    @staticmethod
    def random(reach: int, allows_fixed_points: bool = True) -> 'Permutation':
        """Returns a full random permutation of length `reach`.
        If `allows_fixed_points` is False, the function shuffles the set of fixed points to
        reduce them into a cycle. If the number of fixed points reached 1, another random
        permutation is generated instead.
        """
        keys = list(range(0, reach))
        values = keys[:]
        random.shuffle(values)
        m = dict(zip(keys, values))
        p = Permutation.from_map(m)
        if not allows_fixed_points:
            while p.fixed_points > 0:
                if p.fixed_points > 1:
                    k = [pt[0] for pt in p.get_fixed_points()]
                    v = k[:]
                    random.shuffle(v)
                    m.update(dict(zip(k, v)))
                    p = Permutation.from_map(m)
                else:
                    p = Permutation.random(reach, allows_fixed_points=allows_fixed_points)

        return p

    @staticmethod
    def identity(reach: int) -> 'Permutation':
        """Returns the identity permutation of `reach` size
        """
        with Permutation(reach=reach) as ret:
            for inx in range(0, reach):
                ret.add_cycle(inx)

        return ret

    def canonical(self) -> Representation:
        """Returns the canonical representation of a permutation
        as a product of cycles that obeys the following constraints:

        1. in each cycle the largest element is listed first

        2. the cycles are sorted in ascending order of their first element
        """
        canonical_rep = []
        for cycle in self._representation:
            canonical_cycle = []
            # get the largest element
            lrg = max(cycle)
            pos = cycle.index(lrg)
            elmt = lrg
            while True:
                canonical_cycle.append(elmt)
                elmt = self._find_next(pos, cycle)
                if elmt == lrg:
                    break
                else:
                    pos = cycle.index(elmt)

            canonical_rep.append(tuple(canonical_cycle))

        canonical_rep = tuple(sorted(canonical_rep, key=lambda a: a[0], reverse=True))
        return canonical_rep

    def __iter__(self) -> Iterator[Cycle]:
        """Returns an iterator on all the cycles
        """
        return self._iter_cycles(1, None)

    def __enter__(self) -> 'Permutation':
        """Context manager interface for entering the mutable building mode
        """
        return self._enter_building_mode()

    def __exit__(self, *_):
        """Context manager interface for exiting the mutable building mode
        """
        self._exit_building_mode()

    def __invert__(self) -> 'Permutation':
        """Inverse permutation operator `~`
        """
        return self.invert()

    def __mul__(self, other: 'Permutation') -> 'Permutation':
        """Composition operator `*`
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f'You can compose (multiply) only operands of {self.__class__} type')

        return self.compose(self, other)

    def __pow__(self, power: int, modulo=None) -> 'Permutation':
        """Iterated composition operator `**`
        """
        return self.iterated(power)

    def __eq__(self, other: 'Permutation') -> bool:
        """Equality operator `==`.
        Two permutations are equal if and only if their `canonical`representations are identical.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f'You can compare only operands of {self.__class__} type')

        return other.canonical() == self.canonical()

    # PROTECTED INTERFACE
    def _validate(self, representation: Representation) -> Union[int, None]:
        """Applies rules to a representation:

        1. must be a tuple of tuples of integers
        2. No tuple can be empty
        3. The integers must form a sequence from 0 up to reach as in range(0, reach)

        May throw TypeError, ValueError or PermutationStructuralError (subtypes ValueError)
        """
        indexes = list()
        if not isinstance(representation, tuple):
            raise TypeError('Given representation must be a tuple')

        if len(representation) <= 0:
            raise ValueError('Given representation must not be empty')

        for cycle in representation:
            if not isinstance(cycle, tuple):
                raise TypeError('Given representation must contain only tuples')

            if len(cycle) <= 0:
                raise ValueError('At least one cycle is empty')

            for index in cycle:
                if not isinstance(index, int):
                    raise TypeError('A cycle must contain only integers')
                indexes.append(index)

        if self._reach == 0:
            reach = len(indexes)
        else:
            reach = self._reach

        if tuple(sorted(indexes)) != tuple(range(0, reach)):
            raise PermutationStructuralError('Missing or duplicate index in given representation')

        return len(indexes)

    def _find_element(self, element: int) -> Tuple[Cycle, int]:
        """Finds an element in its cycle
        return cycle, position of <element> in the cycle
        """
        for cycle in self._representation:
            if element in cycle:
                return cycle, cycle.index(element)

        raise IndexError(f'{element} was not found in {self._representation}')

    @staticmethod
    def _find_next(position: int, cycle: Cycle) -> int:
        """Returns the next element from a given position in a cycle
        """
        if position < len(cycle) - 1:
            return cycle[position + 1]
        elif position == len(cycle) - 1:
            return cycle[0]
        else:
            raise IndexError(f'{position} exceeds cycle length {len(cycle)} or is negative')

    def _iter_cycles(self, min_length: int, max_length: Union[int, None]) -> Iterator[Cycle]:
        """Iterator helper. Returns an iterator for all the cycles whose length matches the bounds
        given as parameters. If <max_length> is None, no upper bound is checked
        """
        assert min_length <= max_length if max_length is not None else True
        for cycle in self._representation:
            if max_length is not None:
                if min_length <= len(cycle) <= max_length:
                    yield cycle
            else:
                if len(cycle) >= min_length:
                    yield cycle

    def _enter_building_mode(self):
        self._representation = None
        setattr(self, '_temp_representation', [])

        def add_cycle(*args):
            getattr(self, '_temp_representation').append(tuple(args))

        self.__setattr__('_temp_method', self.add_cycle)
        self.__setattr__(add_cycle.__name__, add_cycle)

        return self

    def _exit_building_mode(self):
        self.__setattr__(self.add_cycle.__name__, self.__getattribute__('_temp_method'))
        self.__delattr__('_temp_method')

        _representation = tuple(getattr(self, '_temp_representation'))
        try:
            self._validate(_representation)
            self._representation = _representation
        except ValueError or TypeError:
            raise
        finally:
            self.__delattr__('_temp_representation')
