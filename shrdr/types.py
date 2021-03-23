"""Module with shrdr template type mappings."""

capacity_types_lookup = {
    'int16': 'CapInt16',
    'int32': 'CapInt32',
    'int64': 'CapInt64',
    'float32': 'CapFloat32',
    'float64': 'CapFloat64',
}

arc_index_types_lookup = {
    'uint32': 'ArcIdxUInt32',
    'uint64': 'ArcIdxUInt64',
}

node_index_types_lookup = {
    'uint32': 'NodeIdxUInt32',
    'uint64': 'NodeIdxUInt64',
}
