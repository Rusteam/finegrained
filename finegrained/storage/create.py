"""Create tables/collections in a DB.
"""
from pathlib import Path
from typing import Any, Dict, List, Union

import fire

from finegrained.utils.milvus import MilvusClient
from finegrained.utils.os_utils import read_file_config


def milvus(collection_name: str, fields: Union[List[Dict[str, Any]], str]):
    if isinstance(fields, str) and Path(fields).exists():
        fields = read_file_config(fields, section=collection_name)
    milvus = MilvusClient()
    milvus.create_collection("test_milvus", fields)


if __name__ == "__main__":
    fire.Fire({"milvus": milvus})
