"""Defined type annotations.
"""
from typing import Dict, List, Tuple, Union

LIST_STR = List[str]
LIST_STR_STR = Union[LIST_STR, str]
LIST_DICT = List[Dict]
LIST_TYPE = Union[list, tuple]

DICT_STR_FLOAT = Dict[str, float]
DICT_STR_STR = Dict[str, str]

IMAGE_SIZE = Union[int, Tuple[int, int]]
