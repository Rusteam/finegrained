"""Utilities to work with Milvus vector search.
"""
from typing import Any, Dict, List

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


class MilvusClient:
    """A class to connect to Milvus and perform CRUD ops."""

    def __init__(self, **kwargs):
        connections.connect(**kwargs)
        self.collections = {}

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name!r} does not exist.")
        return self.collections[collection_name]

    @staticmethod
    def _convert_dtype(field: Dict[str, Any]):
        if "dtype" in field:
            field.update({"dtype": _to_milvus_datatype(field["dtype"])})

    def create_collection(
        self, name: str, fields: List[Dict[str, Any]], description=None
    ):
        """Create a new collection."""
        _ = list(map(self._convert_dtype, fields))
        fields = [FieldSchema(**field) for field in fields]
        schema = CollectionSchema(fields=fields, description=description)
        collection = Collection(name, schema)
        self.collections.update({name: collection})

    @staticmethod
    def drop_collection(name: str):
        utility.drop_collection(name)

    def build_index(self, field: str, collection_name: str, **kwargs):
        collection = self.get_collection(collection_name)
        collection.create_index(field, kwargs)

    def insert_vectors(self, data: List[List[Any]], collection_name: str):
        collection = self.get_collection(collection_name)
        collection.insert(data)

    def search_vectors(
        self,
        vectors: List[List[float]],
        collection_name: str,
        metric_type="l2",
        top_k=5,
        **kwargs,
    ):
        collection = self.get_collection(collection_name)
        collection.load()
        search_params = {
            "metric_type": metric_type,
            "params": {"nprobe": top_k},
        }
        result = collection.search(vectors, search_params, **kwargs)
        return result


def _to_milvus_datatype(data_type):
    if data_type in [str, np.object, np.str]:
        return DataType.NONE
    elif data_type in [bool]:
        return DataType.BOOL
    elif data_type in [int, np.int64]:
        return DataType.INT64
    elif data_type in [np.int32]:
        return DataType.INT32
    elif data_type in [float, np.float64]:
        return DataType.DOUBLE
    elif data_type in [np.float32]:
        return DataType.FLOAT_VECTOR
    else:
        raise ValueError(f"Unsupported data type {data_type}")
