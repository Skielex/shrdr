"""Module containing functions for creating solvers."""
import numpy as np
import numpy.typing as npt

from . import _shrdr
from .types import arc_index_types_lookup, capacity_types_lookup, node_index_types_lookup


def _create_class_name(base_name, capacity_type, arc_index_type, node_index_type):

    try:
        capacity_type = np.dtype(capacity_type).name
    except:
        raise ValueError(f"Invalid capacity type '{capacity_type}'. Must be a valid NumPy dtype.")

    try:
        arc_index_type = np.dtype(arc_index_type).name
    except:
        raise ValueError(f"Invalid arc index type '{arc_index_type}'. Must be a valid NumPy dtype.")

    try:
        node_index_type = np.dtype(node_index_type).name
    except:
        raise ValueError(f"Invalid node index type '{node_index_type}'. Must be a valid NumPy dtype.")

    if capacity_type not in capacity_types_lookup:
        raise ValueError(
            f"Unsupported capacity type '{capacity_type}'. Supported types are: {', '.join(capacity_types_lookup)}")

    if arc_index_type not in arc_index_types_lookup:
        raise ValueError(
            f"Unsupported arc index type '{arc_index_type}'. Supported types are: {', '.join(arc_index_types_lookup)}")

    if node_index_type not in node_index_types_lookup:
        raise ValueError(
            f"Unsupported node index type '{node_index_type}'. Supported types are: {', '.join(node_index_types_lookup)}"
        )

    return base_name + capacity_types_lookup[capacity_type] + arc_index_types_lookup[
        arc_index_type] + node_index_types_lookup[node_index_type]


def qpbo(
    expected_nodes: int = 0,
    expected_pairwise_terms: int = 0,
    expect_nonsubmodular: bool = True,
    capacity_type: npt.DTypeLike = 'int32',
    arc_index_type: npt.DTypeLike = 'uint32',
    node_index_type: npt.DTypeLike = 'uint32',
):
    """Returns a new Qpbo class instance of the specified type."""
    class_name = _create_class_name('Qpbo', capacity_type, arc_index_type, node_index_type)
    class_ctor = getattr(_shrdr, class_name)
    return class_ctor(expected_nodes, expected_pairwise_terms, expect_nonsubmodular)


def parallel_qpbo(
    expected_nodes: int = 0,
    expected_pairwise_terms: int = 0,
    expect_nonsubmodular: bool = True,
    expected_blocks: int = 0,
    capacity_type: npt.DTypeLike = 'int32',
    arc_index_type: npt.DTypeLike = 'uint32',
    node_index_type: npt.DTypeLike = 'uint32',
):
    """Returns a new ParallelQpbo class instance of the specified type."""
    class_name = _create_class_name('ParallelQpbo', capacity_type, arc_index_type, node_index_type)
    class_ctor = getattr(_shrdr, class_name)
    return class_ctor(expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)


def bk(
    expected_nodes: int = 0,
    expected_pairwise_terms: int = 0,
    capacity_type: npt.DTypeLike = 'int32',
    arc_index_type: npt.DTypeLike = 'uint32',
    node_index_type: npt.DTypeLike = 'uint32',
):
    """Returns a new Bk class instance of the specified type."""
    class_name = _create_class_name('Bk', capacity_type, arc_index_type, node_index_type)
    class_ctor = getattr(_shrdr, class_name)
    return class_ctor(expected_nodes, expected_pairwise_terms)


def parallel_bk(
    expected_nodes: int = 0,
    expected_pairwise_terms: int = 0,
    expected_blocks: int = 0,
    capacity_type: npt.DTypeLike = 'int32',
    arc_index_type: npt.DTypeLike = 'uint32',
    node_index_type: npt.DTypeLike = 'uint32',
):
    """Returns a new ParallelBk class instance of the specified type."""
    class_name = _create_class_name('ParallelBk', capacity_type, arc_index_type, node_index_type)
    class_ctor = getattr(_shrdr, class_name)
    return class_ctor(expected_nodes, expected_pairwise_terms, expected_blocks)
