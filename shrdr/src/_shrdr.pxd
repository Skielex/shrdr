# _shrdr.pxd
# distutils: language = c++

from libc.stdint cimport int8_t, uint16_t, int64_t
from libcpp cimport bool

ctypedef uint16_t BlockIdx
ctypedef int8_t NodeLabel

cdef extern from "core/qpbo.h" namespace "mbk":
    cdef cppclass Qpbo[Cap, Flow, ArcIdx, NodeIdx]:
        Qpbo(size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular)
        NodeIdx add_node(int num)
        void add_unary_term(NodeIdx i, Cap e0, Cap e1)
        void add_pairwise_term(NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11)
        NodeLabel get_label(NodeIdx i)
        Flow compute_twice_energy()
        void solve()
        void compute_weak_persistencies()

cdef extern from "core/parallel_qpbo.h" namespace "mbk":
    cdef cppclass ParallelQpbo[Cap, Flow, ArcIdx, NodeIdx]:
        ParallelQpbo(size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular, size_t expected_blocks)
        NodeIdx add_node(int num, BlockIdx block)
        void add_unary_term(NodeIdx i, Cap e0, Cap e1)
        void add_pairwise_term(NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11)
        void add_pairwise_terms(size_t length, NodeIdx* i, NodeIdx* j, Cap* e00, Cap* e01, Cap* e10, Cap* e11)
        NodeLabel get_label(NodeIdx i)
        Flow compute_twice_energy()
        void solve()
        void compute_weak_persistencies()
        unsigned int get_num_threads()
        void set_num_threads(unsigned int num)
