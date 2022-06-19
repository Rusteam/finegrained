"""Defined type annotations.
"""
from typing import Union, List, Dict

LIST_STR = List[str]
LIST_STR_STR = Union[LIST_STR, str]
LIST_DICT = List[Dict]
LIST_TYPE = Union[list, tuple]

DICT_STR_FLOAT = Dict[str, float]
