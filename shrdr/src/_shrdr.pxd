# _shrdr.pxd
# distutils: language = c++

from libc.stdint cimport int8_t, uint16_t, int64_t
from libcpp cimport bool

ctypedef uint16_t BlockIdx
# ctypedef int8_t NodeLabel

cdef extern from "core/util.h" namespace "shrdr":
    ctypedef enum NodeLabel:
        SOURCE = 0
        SINK = 1
        UNKNOWN = -1

cdef extern from "core/qpbo.h" namespace "shrdr":
    cdef cppclass Qpbo[Cap, Flow, ArcIdx, NodeIdx]:
        Qpbo(size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular)
        NodeIdx add_node(int num)
        void add_unary_term(NodeIdx i, Cap e0, Cap e1)
        void add_pairwise_term(NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11)
        NodeLabel get_label(NodeIdx i)
        Flow compute_twice_energy()
        void solve()
        void compute_weak_persistencies()

cdef extern from "core/parallel_qpbo.h" namespace "shrdr":
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

cdef extern from "core/bk.h" namespace "shrdr":
    cdef cppclass Bk[Cap, Term, Flow, ArcIdx, NodeIdx]:
        Bk(size_t expected_nodes, size_t expected_pairwise_terms)
        NodeIdx add_node(int num)
        void add_tweight(NodeIdx i, Term cap_source, Term cap_sink)
        void add_edge(NodeIdx i, NodeIdx j, Cap cap, Cap rev_cap, bool merge_duplicates)
        Flow maxflow(bool reuse_trees)
        Flow get_maxflow()
        NodeLabel what_segment(NodeIdx i, NodeLabel default_segment) const
        void mark_node(NodeIdx i)
