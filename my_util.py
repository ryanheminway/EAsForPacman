"""
Utility functions for all projects.
"""
import math
from typing import Sequence, TypeVar, List, Callable, Iterable, Tuple, Collection, Dict


def getIndexOfMax(values: Sequence, default=-1):
    return max(range(len(values)), key=values.__getitem__, default=default)


def getIndexOfMin(values: Sequence, default=-1):
    return min(range(len(values)), key=values.__getitem__, default=default)


T = TypeVar('T')

def all_argmax(f: Callable[[T], float],
               domain: Iterable[T]) -> Tuple[List[T], float]:
    """
    Returns:
        arg_maxima: all the values in the domain such that f(x) is the max
        max: the max value of over the domain
    """
    max_y = -math.inf
    args = []
    for x in domain:
        y = f(x)
        if y > max_y:
            max_y = y
            args = [x]
        elif y == max_y:
            args.append(x)
    return args, max_y


K = TypeVar('K')
V = TypeVar('V')

def dict_argmax(d: Dict[K, V]) -> Tuple[List[K], V]:
    """ Return the max value in the dict and all the keys whose value is the max """
    return all_argmax(d.__getitem__, d.keys())
