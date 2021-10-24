# distutils: language = c++

from libc.stdint cimport int8_t, int16_t, int32_t, uint32_t, int64_t, uint64_t
from libcpp cimport bool
from .src._shrdr cimport BlockIdx, NodeLabel, Qpbo, ParallelQpbo, Bk, ParallelBk

cimport cython

ctypedef int16_t CapInt16
ctypedef int32_t CapInt32
ctypedef int64_t CapInt64
ctypedef float CapFloat32
ctypedef double CapFloat64
ctypedef uint32_t ArcIdxUInt32
ctypedef uint64_t ArcIdxUInt64
ctypedef uint32_t NodeIdxUInt32
ctypedef uint64_t NodeIdxUInt64
ctypedef int64_t FlowInt
ctypedef double FlowFloat


cdef class QpboCapInt16ArcIdxUInt32NodeIdxUInt32:
    cdef Qpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt16 e0, CapInt16 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt16ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt16 e0, CapInt16 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt16ArcIdxUInt32NodeIdxUInt32:
    cdef Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt16ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt16ArcIdxUInt32NodeIdxUInt64:
    cdef Qpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt16 e0, CapInt16 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt16ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt16 e0, CapInt16 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt16ArcIdxUInt32NodeIdxUInt64:
    cdef Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt16ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt16ArcIdxUInt64NodeIdxUInt32:
    cdef Qpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt16 e0, CapInt16 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt16ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt16 e0, CapInt16 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt16ArcIdxUInt64NodeIdxUInt32:
    cdef Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt16ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt16ArcIdxUInt64NodeIdxUInt64:
    cdef Qpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt16 e0, CapInt16 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt16ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt16 e0, CapInt16 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt16[::1] e0, CapInt16[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 e00, CapInt16 e01, CapInt16 e10, CapInt16 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] e00, CapInt16[::1] e01, CapInt16[::1] e10, CapInt16[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt16ArcIdxUInt64NodeIdxUInt64:
    cdef Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt16ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt16, CapInt16, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt16 cap_source, CapInt16 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt16[::1] cap_source, CapInt16[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt16 cap, CapInt16 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt16[::1] cap, CapInt16[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt32ArcIdxUInt32NodeIdxUInt32:
    cdef Qpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt32 e0, CapInt32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt32ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt32 e0, CapInt32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt32ArcIdxUInt32NodeIdxUInt32:
    cdef Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt32ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt32ArcIdxUInt32NodeIdxUInt64:
    cdef Qpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt32 e0, CapInt32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt32ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt32 e0, CapInt32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt32ArcIdxUInt32NodeIdxUInt64:
    cdef Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt32ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt32ArcIdxUInt64NodeIdxUInt32:
    cdef Qpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt32 e0, CapInt32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt32ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt32 e0, CapInt32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt32ArcIdxUInt64NodeIdxUInt32:
    cdef Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt32ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt32ArcIdxUInt64NodeIdxUInt64:
    cdef Qpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt32 e0, CapInt32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt32ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt32 e0, CapInt32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt32[::1] e0, CapInt32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 e00, CapInt32 e01, CapInt32 e10, CapInt32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] e00, CapInt32[::1] e01, CapInt32[::1] e10, CapInt32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt32ArcIdxUInt64NodeIdxUInt64:
    cdef Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt32ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt32, CapInt32, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt32 cap_source, CapInt32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt32[::1] cap_source, CapInt32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt32 cap, CapInt32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt32[::1] cap, CapInt32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt64ArcIdxUInt32NodeIdxUInt32:
    cdef Qpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt64 e0, CapInt64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt64ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt64 e0, CapInt64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt64ArcIdxUInt32NodeIdxUInt32:
    cdef Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt64ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt64ArcIdxUInt32NodeIdxUInt64:
    cdef Qpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt64 e0, CapInt64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt64ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt64 e0, CapInt64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt64ArcIdxUInt32NodeIdxUInt64:
    cdef Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt64ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt64ArcIdxUInt64NodeIdxUInt32:
    cdef Qpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt64 e0, CapInt64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt64ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapInt64 e0, CapInt64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt64ArcIdxUInt64NodeIdxUInt32:
    cdef Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt64ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapInt64ArcIdxUInt64NodeIdxUInt64:
    cdef Qpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt64 e0, CapInt64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapInt64ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapInt64 e0, CapInt64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapInt64[::1] e0, CapInt64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 e00, CapInt64 e01, CapInt64 e10, CapInt64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] e00, CapInt64[::1] e01, CapInt64[::1] e10, CapInt64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapInt64ArcIdxUInt64NodeIdxUInt64:
    cdef Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapInt64ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapInt64, CapInt64, FlowInt, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapInt64 cap_source, CapInt64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapInt64[::1] cap_source, CapInt64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapInt64 cap, CapInt64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapInt64[::1] cap, CapInt64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat32ArcIdxUInt32NodeIdxUInt32:
    cdef Qpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat32 e0, CapFloat32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat32ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat32 e0, CapFloat32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat32ArcIdxUInt32NodeIdxUInt32:
    cdef Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat32ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat32ArcIdxUInt32NodeIdxUInt64:
    cdef Qpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat32 e0, CapFloat32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat32ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat32 e0, CapFloat32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat32ArcIdxUInt32NodeIdxUInt64:
    cdef Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat32ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat32ArcIdxUInt64NodeIdxUInt32:
    cdef Qpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat32 e0, CapFloat32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat32ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat32 e0, CapFloat32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat32ArcIdxUInt64NodeIdxUInt32:
    cdef Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat32ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat32ArcIdxUInt64NodeIdxUInt64:
    cdef Qpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat32 e0, CapFloat32 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat32ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat32 e0, CapFloat32 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat32[::1] e0, CapFloat32[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 e00, CapFloat32 e01, CapFloat32 e10, CapFloat32 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] e00, CapFloat32[::1] e01, CapFloat32[::1] e10, CapFloat32[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat32ArcIdxUInt64NodeIdxUInt64:
    cdef Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat32ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat32, CapFloat32, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat32 cap_source, CapFloat32 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat32[::1] cap_source, CapFloat32[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat32 cap, CapFloat32 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat32[::1] cap, CapFloat32[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat64ArcIdxUInt32NodeIdxUInt32:
    cdef Qpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat64 e0, CapFloat64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat64ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat64 e0, CapFloat64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat64ArcIdxUInt32NodeIdxUInt32:
    cdef Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat64ArcIdxUInt32NodeIdxUInt32:
    cdef ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat64ArcIdxUInt32NodeIdxUInt64:
    cdef Qpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat64 e0, CapFloat64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat64ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat64 e0, CapFloat64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat64ArcIdxUInt32NodeIdxUInt64:
    cdef Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat64ArcIdxUInt32NodeIdxUInt64:
    cdef ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt32, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat64ArcIdxUInt64NodeIdxUInt32:
    cdef Qpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat64 e0, CapFloat64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat64ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt32 i, CapFloat64 e0, CapFloat64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt32[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt32 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat64ArcIdxUInt64NodeIdxUInt32:
    cdef Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt32 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat64ArcIdxUInt64NodeIdxUInt32:
    cdef ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt32](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt32 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt32[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt32 i, NodeIdxUInt32 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt32[::1] i, NodeIdxUInt32[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt32 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)


cdef class QpboCapFloat64ArcIdxUInt64NodeIdxUInt64:
    cdef Qpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_qpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True):
        self.c_qpbo = new Qpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular)

    def __dealloc__(self):
        del self.c_qpbo

    def add_node(self, size_t num):
        return self.c_qpbo.add_node(num)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat64 e0, CapFloat64 e1):
        self.c_qpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_qpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_qpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_qpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_qpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_qpbo.compute_twice_energy()

    def solve(self):
        self.c_qpbo.solve()

    def compute_weak_persistencies(self):
        self.c_qpbo.compute_weak_persistencies()


cdef class ParallelQpboCapFloat64ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_pqpbo

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, bool expect_nonsubmodular=True, size_t expected_blocks=0):
        self.c_pqpbo = new ParallelQpbo[CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expect_nonsubmodular, expected_blocks)

    def __dealloc__(self):
        del self.c_pqpbo

    def add_node(self, size_t num, BlockIdx block=0):
        return self.c_pqpbo.add_node(num, block)

    def add_unary_term(self, NodeIdxUInt64 i, CapFloat64 e0, CapFloat64 e1):
        self.c_pqpbo.add_unary_term(i, e0, e1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_unary_terms(self, NodeIdxUInt64[::1] i, CapFloat64[::1] e0, CapFloat64[::1] e1):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == e0.shape[0] == e1.shape[0]

        for n in range(length):
            self.c_pqpbo.add_unary_term(i[n], e0[n], e1[n])

    def add_pairwise_term(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 e00, CapFloat64 e01, CapFloat64 e10, CapFloat64 e11):
        self.c_pqpbo.add_pairwise_term(i, j, e00, e01, e10, e11)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_pairwise_terms(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        for n in range(length):
            self.c_pqpbo.add_pairwise_term(i[n], j[n], e00[n], e01[n], e10[n], e11[n])

    def add_pairwise_terms_parallel(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] e00, CapFloat64[::1] e01, CapFloat64[::1] e10, CapFloat64[::1] e11):
        cdef Py_ssize_t length = i.shape[0]
        assert i.shape[0] == j.shape[0] == e00.shape[0] == e01.shape[0] == e10.shape[0] == e11.shape[0]

        self.c_pqpbo.add_pairwise_terms(length, &i[0], &j[0], &e00[0], &e01[0], &e10[0], &e11[0])

    def get_label(self, NodeIdxUInt64 i):
        return self.c_pqpbo.get_label(i)

    def compute_twice_energy(self):
        return self.c_pqpbo.compute_twice_energy()

    def solve(self):
        self.c_pqpbo.solve()

    def compute_weak_persistencies(self):
        self.c_pqpbo.compute_weak_persistencies()

    def get_num_threads(self):
        return self.c_pqpbo.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pqpbo.set_num_threads(num)


cdef class BkCapFloat64ArcIdxUInt64NodeIdxUInt64:
    cdef Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_bk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0):
        self.c_bk = new Bk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms)

    def __dealloc__(self):
        del self.c_bk

    def add_node(self, size_t num):
        return self.c_bk.add_node(num)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_bk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_bk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_bk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_bk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self, bool reuse_trees=False):
        return self.c_bk.maxflow(reuse_trees)

    def get_maxflow(self):
        return self.c_bk.get_maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_bk.what_segment(i, default_segment)

    def mark_node(self, NodeIdxUInt64 i):
        self.c_bk.mark_node(i)


cdef class ParallelBkCapFloat64ArcIdxUInt64NodeIdxUInt64:
    cdef ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64]* c_pbk

    def __cinit__(self, size_t expected_nodes=0, size_t expected_pairwise_terms=0, size_t expected_blocks=0):
        self.c_pbk = new ParallelBk[CapFloat64, CapFloat64, FlowFloat, ArcIdxUInt64, NodeIdxUInt64](expected_nodes, expected_pairwise_terms, expected_blocks)

    def __dealloc__(self):
        del self.c_pbk

    def add_node(self, size_t num, BlockIdx block):
        return self.c_pbk.add_node(num, block)

    def add_tweight(self, NodeIdxUInt64 i, CapFloat64 cap_source, CapFloat64 cap_sink):
        self.c_pbk.add_tweight(i, cap_source, cap_sink)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_tweights(self, NodeIdxUInt64[::1] i, CapFloat64[::1] cap_source, CapFloat64[::1] cap_sink):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == cap_source.shape[0] == cap_sink.shape[0]

        for n in range(length):
            self.c_pbk.add_tweight(i[n], cap_source[n], cap_sink[n])

    def add_edge(self, NodeIdxUInt64 i, NodeIdxUInt64 j, CapFloat64 cap, CapFloat64 rev_cap, bool merge_duplicates=False):
        self.c_pbk.add_edge(i, j, cap, rev_cap, merge_duplicates)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_edges(self, NodeIdxUInt64[::1] i, NodeIdxUInt64[::1] j, CapFloat64[::1] cap, CapFloat64[::1] rev_cap, bool merge_duplicates=False):
        cdef Py_ssize_t length = i.shape[0]

        assert i.shape[0] == j.shape[0] == cap.shape[0] == rev_cap.shape[0]

        for n in range(length):
            self.c_pbk.add_edge(i[n], j[n], cap[n], rev_cap[n], merge_duplicates)

    def maxflow(self):
        return self.c_pbk.maxflow()

    def what_segment(self, NodeIdxUInt64 i, NodeLabel default_segment = NodeLabel.SOURCE):
        return self.c_pbk.what_segment(i, default_segment)

    def get_num_threads(self):
        return self.c_pbk.get_num_threads()

    def set_num_threads(self, unsigned int num):
        self.c_pbk.set_num_threads(num)
