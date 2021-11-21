#ifndef SHRDR_PARALLEL_QPBO_H__
#define SHRDR_PARALLEL_QPBO_H__

#include <list>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <cinttypes>
#include <cassert>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <numeric>
#include <vector>

#include "robin_hood.h"

#include "util.h"

namespace shrdr {

using BlockIdx = uint16_t; // We assume 65536 is enough blocks
using BoundaryKey = uint32_t; // Must be 2 x sizeof(BlockIdx)
using Time = uint32_t;
using Dist = uint16_t;

static_assert(sizeof(BoundaryKey) == 2 * sizeof(BlockIdx),
    "BoundaryKey must be double the size of BlockIdx");

template <
    class Cap,
    class Flow = typename std::conditional<std::is_floating_point<Cap>::value, double, int64_t>::type,
    class ArcIdx = uint32_t,
    class NodeIdx = uint32_t
>
class ParallelQpbo {
    static_assert(std::is_integral<ArcIdx>::value, "ArcIdx must be an integer type");
    static_assert(std::is_integral<NodeIdx>::value, "NodeIdx must be an integer type");
    static_assert(std::is_signed<Cap>::value, "Cap must be a signed type");

    // Forward decls.
    struct Node;
    struct Arc;
    struct BoundarySegment;
    struct QpboBlock;

public:
    static const NodeIdx INVALID_NODE = ~NodeIdx(0); // -1 for signed type, max. value for unsigned type
    static const ArcIdx INVALID_ARC = ~ArcIdx(0); // -1 for signed type, max. value for unsigned type
    static const ArcIdx TERMINAL_ARC = INVALID_ARC - 1;
    static const ArcIdx ORPHAN_ARC = INVALID_ARC - 2;

    ParallelQpbo(size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular,
        size_t expected_blocks);

    NodeIdx add_node(size_t num = 1, BlockIdx block = 0);

    void add_unary_term(NodeIdx i, Cap e0, Cap e1);

    void add_pairwise_term(NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11);
    void add_pairwise_terms(size_t length, NodeIdx* is, NodeIdx* js, Cap* e00s, Cap* e01s, Cap* e10s, Cap* e11s);

    NodeLabel get_label(NodeIdx i) const;
    NodeLabel what_segment(NodeIdx i, NodeLabel default_segment = SOURCE) const;

    std::pair<Cap, Cap> get_twice_unary_term(NodeIdx i) const;
    std::tuple<Cap, Cap, Cap, Cap> get_twice_pairwise_term(ArcIdx term) const;

    std::pair<NodeIdx, NodeIdx> get_arc_ends(ArcIdx ai) const;

    Flow compute_twice_energy() const;
    inline Flow get_flow() const noexcept { return flow; };

    size_t get_primal_node_num() const noexcept { return nodes.size() - (stage > 0) * node_shift; };
    size_t get_total_node_num() const noexcept { return nodes.size(); };
    ArcIdx get_primal_arc_num() const noexcept { return arcs.size() - (stage > 0) * arc_shift; };
    ArcIdx get_total_arc_num() const noexcept { return arcs.size(); };

    inline unsigned int get_num_threads() const noexcept { return num_threads; }
    inline void set_num_threads(unsigned int num) noexcept { num_threads = num; }

    void solve();

    void compute_weak_persistencies();

private:
    std::vector<Node> nodes;
    std::vector<Arc> arcs;

    std::vector<BlockIdx> node_blocks;

    robin_hood::unordered_map<BoundaryKey, ArcIdx> boundary_arcs;
    robin_hood::unordered_set<BoundaryKey> non_submodular_boundaries;
    std::list<BoundarySegment> boundary_segments;
    std::vector<BlockIdx> block_idxs;

    std::vector<QpboBlock> blocks;

    std::vector<NodeIdx> next_block_node;
    std::vector<ArcIdx> next_block_arc;

    bool all_submodular;

    unsigned int num_threads;

    Flow zero_energy;
    Flow flow;

    NodeIdx node_shift;
    ArcIdx arc_shift;

    int stage;

    void init_maxflow();
    void prepare_for_second_stage_transform();

    std::tuple<Cap, Cap, Cap, Cap> compute_normal_form_weights(
        Cap e00, Cap e01, Cap e10, Cap e11) const noexcept;

    ArcIdx add_half_edge(NodeIdx from, NodeIdx to, Cap cap);
    ArcIdx set_half_edge(ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap);
    ArcIdx add_half_edge(NodeIdx to, Cap cap);
    ArcIdx set_half_edge(ArcIdx ai, NodeIdx to, Cap cap);
    ArcIdx add_boundary_half_edge(NodeIdx to, Cap cap, BoundaryKey key);
    ArcIdx set_boundary_half_edge(ArcIdx ai, NodeIdx to, Cap cap, BoundaryKey key);

    void build_half_edge(ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap);

    void add_outgoing(NodeIdx from, ArcIdx ai);

    bool should_activate(NodeIdx i, NodeIdx j);

    inline BoundaryKey block_key(BlockIdx i, BlockIdx j) const noexcept;
    inline std::pair<BlockIdx, BlockIdx> blocks_from_key(BoundaryKey key) const noexcept;

    std::tuple<std::list<BoundarySegment>, BlockIdx, BlockIdx, bool> next_boundary_segment_set();
    void unite_blocks(BlockIdx i, BlockIdx j);

    /** Build primal-dual graph construction and add non-submodular edges */
    void transform_to_second_stage(BlockIdx block);

    void register_block_nodes(NodeIdx first, NodeIdx last, BlockIdx block);
    void register_block_arc(ArcIdx i, BlockIdx block);
    ArcIdx register_boundary_arc(ArcIdx i, BoundaryKey key);

    inline ArcIdx sister_idx(ArcIdx a) const noexcept { return a ^ 1; }
    inline Arc &sister(ArcIdx a) { return arcs[sister_idx(a)]; }
    inline const Arc &sister(ArcIdx a) const { return arcs[sister_idx(a)]; }

    inline NodeIdx dual_node(NodeIdx i) const noexcept { return i + node_shift; }
    inline NodeIdx primal_node(NodeIdx i) const noexcept { return i - node_shift; }
    inline NodeIdx mirror_node(NodeIdx i) const noexcept;

    inline ArcIdx dual_arc(ArcIdx i) const noexcept { return i + arc_shift; }
    inline ArcIdx primal_arc(ArcIdx i) const noexcept { return i - arc_shift; }
    inline ArcIdx mirror_arc(ArcIdx i) const noexcept;

    inline bool is_primal_node(NodeIdx i) const noexcept { return i < node_shift; }
    inline bool is_dual_node(NodeIdx i) const noexcept { return i >= node_shift; }
    inline bool is_primal_arc(ArcIdx i) const noexcept { return i < arc_shift; }
    inline bool is_dual_arc(ArcIdx i) const noexcept { return i >= arc_shift; }

    struct BoundarySegment {
        ArcIdx first;
        bool all_submodular;
        BlockIdx i;
        BlockIdx j;
        int32_t potential_activations;
    };

    struct QpboBlock {
        std::vector<Node>& nodes;
        std::vector<Arc>& arcs;

        std::vector<BlockIdx>& node_blocks;

        const NodeIdx& node_shift;
        const ArcIdx& arc_shift;

        bool locked;
        bool initialized;

        Flow flow;

        NodeIdx first_active, last_active;
        std::deque<NodeIdx> orphan_nodes;

        NodeIdx first_node, last_node;
        ArcIdx first_arc, last_arc;

        Time time;

        bool all_submodular;
        int stage;
        
        QpboBlock(std::vector<Node>& nodes, std::vector<Arc>& arcs, std::vector<BlockIdx>& node_blocks,
            const NodeIdx& node_shift, const ArcIdx& arc_shift) :
            nodes(nodes),
            arcs(arcs),
            node_blocks(node_blocks),
            node_shift(node_shift),
            arc_shift(arc_shift),
            locked(false),
            initialized(false),
            flow(0),
            first_active(INVALID_NODE),
            last_active(INVALID_NODE),
            orphan_nodes(),
            first_node(INVALID_NODE),
            last_node(INVALID_NODE),
            first_arc(INVALID_ARC),
            last_arc(INVALID_ARC),
            time(0),
            all_submodular(true),
            stage(0) {}

        Flow maxflow();

        void make_active(NodeIdx i);
        void make_front_orphan(NodeIdx i);
        void make_back_orphan(NodeIdx i);

        NodeIdx next_active();

        void augment(ArcIdx middle);
        Cap tree_bottleneck(NodeIdx start, bool source_tree) const;
        void augment_tree(NodeIdx start, Cap bottleneck, bool source_tree);

        ArcIdx grow_search_tree(NodeIdx start);
        template <bool source> ArcIdx grow_search_tree_impl(NodeIdx start);

        void process_orphan(NodeIdx i);
        template <bool source> void process_orphan_impl(NodeIdx i);

        inline NodeIdx dual_node(NodeIdx i) const noexcept { return i + node_shift; }
        inline NodeIdx primal_node(NodeIdx i) const noexcept { return i - node_shift; }

        inline ArcIdx dual_arc(ArcIdx i) const noexcept { return i + arc_shift; }
        inline ArcIdx primal_arc(ArcIdx i) const noexcept { return i - arc_shift; }

        inline ArcIdx sister_idx(ArcIdx a) const noexcept { return a ^ 1; }
        inline Arc &sister(ArcIdx a) { return arcs[sister_idx(a)]; }
        inline const Arc &sister(ArcIdx a) const { return arcs[sister_idx(a)]; }

        inline ArcIdx sister_or_arc_idx(ArcIdx a, bool sis) const noexcept { return a ^ (ArcIdx)sis; }
        inline Arc& sister_or_arc(ArcIdx a, bool sis) { return arcs[sister_or_arc_idx(a, sis)]; }
        inline const Arc& sister_or_arc(ArcIdx a, bool sis) const { return arcs[sister_or_arc_idx(a, sis)]; }

        inline Node &head_node(const Arc &a) { return nodes[a.head]; }
        inline Node &head_node(ArcIdx a) { return head_node(arcs[a]); }
        inline const Node &head_node(const Arc &a) const { return nodes[a.head]; }
        inline const Node &head_node(ArcIdx a) const { return head_node(arcs[a]); }

        inline bool is_primal_node(NodeIdx i) const noexcept { return i < node_shift; }
        inline bool is_dual_node(NodeIdx i) const noexcept { return i >= node_shift; }
        inline bool is_primal_arc(ArcIdx i) const noexcept { return i < arc_shift; }
        inline bool is_dual_arc(ArcIdx i) const noexcept { return i >= arc_shift; }
    };

#pragma pack (1)
    struct SHRDR_PACKED Node {
        ArcIdx first; // First out-going arc.
        NodeIdx next_active; // Index of next active node (or itself if this is the last one)

        union {
            struct SHRDR_PACKED {
                ArcIdx parent; // Arc to parent node in search tree
                Time timestamp; // Timestamp showing when dist was computed.
                Dist dist; // Distance to terminal.
            };
            struct SHRDR_PACKED {
                ArcIdx dfs_current;
                NodeIdx dfs_parent;
                int32_t region;
            };
        };

        Cap tr_cap; // If tr_cap > 0 then tr_cap is residual capacity of the arc SOURCE->node.
                     // Otherwise         -tr_cap is residual capacity of the arc node->SINK.

        bool is_sink : 1;	// flag showing if the node is in the source or sink tree (if parent!=NULL)
        NodeLabel label : 2;

        Node() :
            first(INVALID_ARC),
            parent(INVALID_ARC),
            next_active(INVALID_NODE),
            timestamp(0),
            dist(0),
            tr_cap(0),
            is_sink(false),
            label(UNKNOWN) {}
    };

    struct SHRDR_PACKED Arc {
        NodeIdx head; // Node this arc points to.
        ArcIdx next; // Next arc with the same originating node

        Cap r_cap; // Residual capacity

        Arc() {} // Do nothing - any initialization results in a massive performance hit

        Arc(NodeIdx _head, ArcIdx _next, Cap _r_cap) :
            head(_head),
            next(_next),
            r_cap(_r_cap) {}
    };
#pragma pack ()
};

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const NodeIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::INVALID_NODE;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::INVALID_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::TERMINAL_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::ORPHAN_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::ParallelQpbo(
    size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular, 
    size_t expected_blocks) :
    nodes(),
    arcs(),
    node_blocks(),
    boundary_arcs(),
    non_submodular_boundaries(),
    boundary_segments(),
    block_idxs(),
    blocks(),
    next_block_node(),
    next_block_arc(),
    all_submodular(true),
    num_threads(std::thread::hardware_concurrency()),
    zero_energy(0),
    flow(0),
    node_shift(expected_nodes),
    arc_shift(0),
    stage(0)
{
    nodes.reserve((expect_nonsubmodular ? 2 : 1) * expected_nodes);
    node_blocks.reserve((expect_nonsubmodular ? 2 : 1) * expected_nodes);
    next_block_node.reserve(expected_nodes);
    arcs.reserve((expect_nonsubmodular ? 4 : 2) * expected_pairwise_terms);
    next_block_arc.reserve(2 * expected_pairwise_terms);
    blocks.reserve(expected_blocks);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_node(size_t num, BlockIdx block)
{
    assert(nodes.size() == node_blocks.size());
    NodeIdx crnt = nodes.size();

#ifndef SHRDR_NO_OVERFLOW_CHECKS
    if (crnt > std::numeric_limits<NodeIdx>::max() - num) {
        throw std::overflow_error("Node count exceeds capacity of index type. "
            "Please increase capacity of NodeIdx type.");
    }
#endif

    if (crnt + num > node_shift) {
        if (!all_submodular) {
            // Since non-submodular terms have been added, we first need to to update all head indices for
            // all arcs between the primal and dual graph
            for (auto& a : arcs) {
                if (is_dual_node(a.head)) {
                    a.head += num;
                }
            }
        }
        node_shift += num;
    }

    nodes.resize(crnt + num);
    if (block >= blocks.size()) {
        // We assume that we always have blocks 0,1,2,...,N
        for (int b = blocks.size(); b <= block; ++b) {
            blocks.emplace_back(nodes, arcs, node_blocks, node_shift, arc_shift);
            block_idxs.push_back(b);
        }
    }
    // Register nodes with block
    node_blocks.resize(crnt + num, block);
    next_block_node.resize(crnt + num, INVALID_NODE);
    register_block_nodes(crnt, crnt + num - 1, block);

    return crnt;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_unary_term(NodeIdx i, Cap e0, Cap e1)
{
    nodes[i].tr_cap += e1 - e0;
    zero_energy += e0;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
SHRDR_ALWAYS_INLINE inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_pairwise_term(
    NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11)
{
#ifndef SHRDR_NO_OVERFLOW_CHECKS
    if (arcs.size() > std::numeric_limits<ArcIdx>::max() - 2) {
        throw std::overflow_error("Arc count exceeds capacity of index type. "
            "Please increase capacity of ArcIdx type.");
    }
#endif
    Cap ci, cj, cij, cji;
    BlockIdx bi = node_blocks[i];
    BlockIdx bj = node_blocks[j];
    const bool same_block = bi == bj;
    BoundaryKey key;
    if (!same_block) {
        // NOTE: Only pre-computing the key if needed results in a significant performance boost
        key = block_key(bi, bj);
    }
    next_block_arc.resize(arc_shift + 2, INVALID_ARC);
    if (e00 + e11 <= e01 + e10) {
        // Cap is submodular
        std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e00, e01, e10, e11);

        if (same_block) {
            ArcIdx a1 = add_half_edge(i, j, cij);
            add_half_edge(j, i, cji);
            register_block_arc(a1, bi);
        } else {
            add_boundary_half_edge(j, cij, key);
            add_half_edge(i, cji);
        }
    } else {
        // Cap is *not* submodular
        all_submodular = false;
        // Note that energy coefs. are switched!
        std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e01, e00, e11, e10);

        // We only set head and r_cap here and update next when we move to the second stage
        // Only mark block as having non-submodular terms if the term is internal for the block
        // Otherwise, we'll fix it later when we add the boundary segment
        if (same_block) {
            ArcIdx a1 = add_half_edge(dual_node(j), cij);
            blocks[bi].all_submodular = false;
            register_block_arc(a1, bi);
        } else {
            add_boundary_half_edge(dual_node(j), cij, key);
            non_submodular_boundaries.insert(key);
        }
        add_half_edge(i, cji);
    }

    // Terminal arcs
    nodes[i].tr_cap += ci;
    nodes[j].tr_cap += cj;

    zero_energy += e00;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_pairwise_terms(
    size_t length, NodeIdx is[], NodeIdx js[], Cap e00s[], Cap e01s[], Cap e10s[], Cap e11s[])
{
    // Make space for new arcs.
    size_t initial_arc_count = arcs.size();
    size_t new_arc_count = length * 2;
    arcs.resize(initial_arc_count + new_arc_count);
    arc_shift += new_arc_count;
    next_block_arc.resize(arc_shift + 2, INVALID_ARC);

    std::vector<std::thread> threads;
    std::mutex arcs_lock;
    std::mutex nodes_lock;
    auto thread_count = std::min<unsigned int>(num_threads, blocks.size());
    auto blocks_per_thread = (BlockIdx) std::ceil(((float) blocks.size()) / thread_count);

    for (unsigned int t = 0; t < thread_count; ++t) {
        threads.emplace_back([&](int id) {
            ArcIdx ai = initial_arc_count;
            BlockIdx start_idx = blocks_per_thread * id;
            BlockIdx stop_idx = start_idx + blocks_per_thread;

            Cap ci, cj, cij, cji;

            for (size_t n = 0; n < length; n++) {
                NodeIdx i = is[n];
                BlockIdx bi = node_blocks[i];

                if (bi < start_idx || bi >= stop_idx) {
                    ai += 2;
                    continue;
                }

                NodeIdx j = js[n];
                Cap e00 = e00s[n], e01 = e01s[n], e10 = e10s[n], e11 = e11s[n];
                BlockIdx bj = node_blocks[j];
                const bool same_block = bi == bj;
                BoundaryKey key;
                if (!same_block) {
                    // NOTE: Only pre-computing the key if needed results in a significant performance boost
                    key = block_key(bi, bj);
                }
                if (e00 + e11 <= e01 + e10) {
                    // Cap is submodular
                    std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e00, e01, e10, e11);

                    if (same_block) {
                        ArcIdx a1 = set_half_edge(ai++, i, j, cij);
                        set_half_edge(ai++, j, i, cji);
                        register_block_arc(a1, bi);
                    } else {
                        arcs_lock.lock();
                        set_boundary_half_edge(ai++, j, cij, key);
                        arcs_lock.unlock();
                        set_half_edge(ai++, i, cji);
                    }
                } else {
                    // Cap is *not* submodular
                    all_submodular = false;
                    // Note that energy coefs. are switched!
                    std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e01, e00, e11, e10);

                    // We only set head and r_cap here and update next when we move to the second stage
                    // Only mark block as having non-submodular terms if the term is internal for the block
                    // Otherwise, we'll fix it later when we add the boundary segment
                    if (same_block) {
                        ArcIdx a1 = set_half_edge(ai++, dual_node(j), cij);
                        blocks[bi].all_submodular = false;
                        register_block_arc(a1, bi);
                    } else {
                        arcs_lock.lock();
                        set_boundary_half_edge(ai++, dual_node(j), cij, key);
                        non_submodular_boundaries.insert(key);
                        arcs_lock.unlock();
                    }
                    set_half_edge(ai++, i, cji);
                }

                // Terminal arcs
                if (ci != 0 || cj != 0 || e00 != 0) {
                nodes_lock.lock();
                nodes[i].tr_cap += ci;
                nodes[j].tr_cap += cj;

                zero_energy += e00;
                nodes_lock.unlock();
            }
            }
        }, t);
    }
    for (auto &th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeLabel ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::get_label(NodeIdx i) const
{
    assert(0 <= i && is_primal_node(i));
    return nodes[i].label;
    }

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeLabel ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::what_segment(NodeIdx i, NodeLabel default_segment) const
{
    assert(0 <= i && i < nodes.size());
    if (nodes[i].parent != INVALID_ARC) {
        return (nodes[i].is_sink) ? SINK : SOURCE;
    } else {
        return default_segment;
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::pair<Cap, Cap> ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::get_twice_unary_term(NodeIdx i) const
{
    assert(0 <= i && is_primal_node(i));
    Cap twice_cap;
    if (stage == 0) {
        twice_cap = 2 * nodes[i].tr_cap;
    } else {
        twice_cap = nodes[i].tr_cap - nodes[dual_node(i)].tr_cap;
    }
    return std::make_pair(0, twice_cap);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::tuple<Cap, Cap, Cap, Cap> ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::get_twice_pairwise_term(
    ArcIdx term) const
{
    ArcIdx a2, a1;

    // Select half edges such that a1 points out of a primal node
    // NOTE: Since we have two arcs per term, the arc index corresponds to 2 * term
    a1 = 2 * term;
    assert(0 <= a1 && is_primal_arc(a1));
    if (is_primal_node(sister(a1).head)) {
        a2 = dual_arc(a1);
    } else {
        a2 = sister_idx(a1);
        a1 = dual_arc(a2);
    }

    Cap e1, e2;
    if (stage == 0) {
        e1 = 2 * arcs[a1].r_cap;
        e2 = 2 * sister(a1).r_cap;
    } else {
        e1 = arcs[a1].r_cap + arcs[a2].r_cap;
        e2 = sister(a1).r_cap + sister(a2).r_cap;
    }

    if (is_primal_node(arcs[a1].head)) {
        // Primal-to-primal means we have a submodular term
        return std::make_tuple(0, e1, e2, 0);
    } else {
        // Primal-to-dual means we have a non-submodular term
        return std::make_tuple(e1, 0, 0, e2);
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::pair<NodeIdx, NodeIdx> ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::get_arc_ends(ArcIdx ai) const
{
    assert(0 <= ai && ai <= arcs.size());
    return std::make_pair(arcs[ai].head, sister(ai).head);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Flow ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_twice_energy() const
{
    // TODO: This way of computing the energy is not stable if we use inifite floating point cost
    Flow energy = 2 * zero_energy;

    // Unary terms
    Cap e1[2];
    const NodeIdx num_nodes = get_primal_node_num();
    for (NodeIdx i = 0; i < num_nodes; ++i) {
        std::tie(e1[0], e1[1]) = get_twice_unary_term(i);
        uint8_t xi = get_label(i) == SINK;
        energy += e1[xi] - e1[0];
    }

    // Binary terms
    Cap e2[2][2];
    NodeIdx i, j;
    assert(get_primal_arc_num() % 2 == 0);
    const ArcIdx num_terms = get_primal_arc_num() / 2;
    for (ArcIdx term = 0; term < num_terms; ++term) {
        std::tie(e2[0][0], e2[0][1], e2[1][0], e2[1][1]) = get_twice_pairwise_term(term);
        std::tie(i, j) = get_arc_ends(2 * term + 1);
        j = is_dual_node(j) ? primal_node(j) : j;
        uint8_t xi = get_label(i) == SINK;
        uint8_t xj = get_label(j) == SINK;
        energy += e2[xi][xj] - e2[0][0];
    }

    return energy;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::solve()
{
    // Initialize
    if (!all_submodular) {
        prepare_for_second_stage_transform();
    }
    init_maxflow();

    std::atomic<BlockIdx> processed_blocks(0);
    std::vector<std::thread> threads;

    // Phase 1: Solve all base blocks
    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&](int id) {
            char thread_name[32];
            std::snprintf(thread_name, sizeof(thread_name), "Phase 1: %d", id);

            const BlockIdx num_blocks = blocks.size();
            BlockIdx crnt = processed_blocks.fetch_add(1);
            while (crnt < num_blocks) {
                auto& b = blocks[crnt];
                b.maxflow();
                if (!b.all_submodular) {
                    transform_to_second_stage(crnt);
                    b.maxflow();
                }
                crnt = processed_blocks.fetch_add(1);
            }
        }, i);
    }
    for (auto &th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Build list of boundary segments
    for (const auto& ba : boundary_arcs) {
        BlockIdx i, j;
        std::tie(i, j) = blocks_from_key(ba.first);
        bool all_submodular = non_submodular_boundaries.find(ba.first) == non_submodular_boundaries.end();
        boundary_segments.push_back({ ba.second, all_submodular, i, j, 0 });
    }

    // Count number of potential activations for each boundary segment and sort
    // Also check if the boundary segment contains arcs from non-submodular terms
    for (auto &bs : boundary_segments) {
        int32_t potential_activations = 0;
        for (ArcIdx a = bs.first; a != INVALID_ARC; a = arcs[a].next) {
            Arc& arc = arcs[a];
            Arc& sister_arc = sister(a);
            if (is_dual_node(arc.head)) {
                bs.all_submodular = false;
                potential_activations++;
            } else if (should_activate(arc.head, sister_arc.head)) {
                potential_activations++;
            }
        }
        bs.potential_activations = potential_activations;
    }

    boundary_segments.sort([](const auto& bs1, const auto& bs2) {
        return bs1.potential_activations > bs2.potential_activations;
    });

    // Phase 2: Merge blocks
    threads.clear();
    std::mutex lock;

    for (unsigned int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&](int id) {
            char thread_name[32];
            std::snprintf(thread_name, sizeof(thread_name), "Phase 2: %d", id);

            BlockIdx crnt, other;
            std::list<BoundarySegment> boundary_set;
            bool submodular_boundary;
            while (true) {
                lock.lock();

                std::tie(boundary_set, crnt, other, submodular_boundary) = next_boundary_segment_set();
                if (boundary_set.empty()) {
                    lock.unlock();
                    break;
                }

                // Lock blocks
                auto& block = blocks[crnt];
                auto& other_block = blocks[other];
                block.locked = true;
                other_block.locked = true;

                lock.unlock();

                // Ensure blocks are in second stage if needed
                if (!submodular_boundary && block.stage == 0 && other_block.stage == 0) {
                    transform_to_second_stage(crnt);
                    transform_to_second_stage(other);
                } else if (block.stage > 0 && other_block.stage == 0) {
                    transform_to_second_stage(other);
                } else if (other_block.stage > 0 && block.stage == 0) {
                    transform_to_second_stage(crnt);
                }

                // Unite blocks
                unite_blocks(crnt, other);
                if (!submodular_boundary) {
                    block.all_submodular = false;
                }

                // Activate boundary arcs
                for (const auto& bs : boundary_set) {
                    ArcIdx next;
                    for (ArcIdx ai = bs.first; ai != INVALID_ARC; ai = next) {
                        assert(ai + 1 == sister_idx(ai));
                        Arc& arc = arcs[ai];
                        Arc& sister_arc = sister(ai);
                        NodeIdx ni = sister_arc.head;
                        NodeIdx nj = arc.head;
                        next = arc.next;
                    
                        const bool activate = should_activate(ni, nj);
                        if (activate) {
                            block.make_active(ni);
                            block.make_active(nj);
                        }
                        add_outgoing(ni, ai);
                        add_outgoing(nj, ai + 1);
                        if (block.stage > 0) {
                            // Also activate mirror node
                            if (activate) {
                                block.make_active(mirror_node(ni));
                                block.make_active(mirror_node(nj));
                            }
                            // Add mirror primal-to-dual copy
                            NodeIdx mi = mirror_node(ni);
                            NodeIdx mj = mirror_node(nj);
                            build_half_edge(dual_arc(ai), mj, mi, arc.r_cap);
                            build_half_edge(dual_arc(ai) + 1, mi, mj, sister_arc.r_cap);
                        } else {
                            // If this block is still in stage one, we need to keep track of added arcs
                            register_block_arc(ai, crnt);
                        }
                    }
                }

                // Compute maxflow
                block.maxflow();

                lock.lock();
                // Finish uniting the blocks
                std::replace(block_idxs.begin(), block_idxs.end(), other, crnt);
                // NOTE: We don't need to unlock the other block since it effectively stops existing
                block.locked = false;
                lock.unlock();
            }
        }, i);
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }


    // Sum up all subgraph flows
    std::sort(block_idxs.begin(), block_idxs.end());
    auto last = std::unique(block_idxs.begin(), block_idxs.end());
    for (auto iter = block_idxs.begin(); iter != last; ++iter) {
        BlockIdx b = *iter;
        if (!all_submodular && blocks[b].stage == 0) {
            // If the block hasn't been transformed yet, now is the time...
            transform_to_second_stage(b);
        }
        flow += blocks[b].flow;
    }

    for (NodeIdx i = 0; i < get_primal_node_num(); i++) {
        NodeLabel primal = what_segment(i);
        if (all_submodular) {
            nodes[i].label = primal;
        } else {
            NodeLabel dual = what_segment(dual_node(i));
            nodes[i].label = (primal != dual) ? primal : UNKNOWN;
        }
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::init_maxflow()
{
    // Initialize all blocks
    for (auto& b : blocks) {
        b.first_active = INVALID_NODE;
        b.last_active = INVALID_NODE;
        b.orphan_nodes.clear();
        b.time = 0;
    }

    // Initialize nodes
    size_t num_nodes = get_primal_node_num();
    for (size_t i = 0; i < num_nodes; ++i) {
        Node& n = nodes[i];
        BlockIdx bi = block_idxs[node_blocks[i]];
        n.next_active = INVALID_NODE;
        n.timestamp = 0;
        if (n.tr_cap != 0) {
            // n is connected to the source or sink
            n.is_sink = n.tr_cap < 0; // negative capacity goes to sink
            n.parent = TERMINAL_ARC;
            n.dist = 1;
            blocks[bi].make_active(i);
        } else {
            n.parent = INVALID_ARC;
        }
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::prepare_for_second_stage_transform()
{
    // If num_nodes < dual_shift, the primal nodes are padded with dummy nodes
    assert(nodes.size() == node_blocks.size());
    const size_t crnt_nodes = nodes.size();
    const size_t crnt_arcs = arcs.size();
    const size_t new_nodes_size = node_shift + crnt_nodes;
    const size_t new_arcs_size = 2 * crnt_arcs;

    nodes.resize(new_nodes_size);
    arcs.resize(new_arcs_size);
    
    // Copy node blocks over
    node_blocks.resize(new_nodes_size);
    std::copy_n(node_blocks.begin(), crnt_nodes, node_blocks.begin() + node_shift);

    stage = 1;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::tuple<Cap, Cap, Cap, Cap> ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_normal_form_weights(
    Cap e00, Cap e01, Cap e10, Cap e11) const noexcept
{
    // The goal is to rewrite the energy terms e00, e10, e10, e11 so:
    //  1. They are in normal form
    //  2. Only e10 and e01 are non-zero
    Cap ci, cj, cij, cji;
    ci = e11 - e00;
    cij = e01 - e00;
    cji = e10 - e11;

    if (cij < 0) {
        ci -= cij;
        cj = cij;
        cji += cij;
        cij = 0;
    } else if (cji < 0) {
        ci += cji;
        cj = -cji;
        cij += cji;
        cji = 0;
    } else {
        cj = 0;
    }

    return std::make_tuple(ci, cj, cij, cji);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_half_edge(NodeIdx from, NodeIdx to, Cap cap)
{
    ArcIdx ai = arc_shift++;
    arcs.emplace_back(to, nodes[from].first, cap);
    nodes[from].first = ai;
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::set_half_edge(ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap)
{
    Arc& a = arcs[ai];
    a.head = to;
    a.next = nodes[from].first;
    a.r_cap = cap;
    nodes[from].first = ai;
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_half_edge(NodeIdx to, Cap cap)
{
    ArcIdx ai = arc_shift++;
    arcs.emplace_back(to, INVALID_ARC, cap);
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::set_half_edge(ArcIdx ai, NodeIdx to, Cap cap)
{
    Arc& a = arcs[ai];
    a.head = to;
    a.next = INVALID_ARC;
    a.r_cap = cap;
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_boundary_half_edge(
    NodeIdx to, Cap cap, BoundaryKey key)
{
    ArcIdx ai = arc_shift++;
    ArcIdx next_in_boundary = register_boundary_arc(ai, key);
    arcs.emplace_back(to, next_in_boundary, cap);
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::set_boundary_half_edge(
    ArcIdx ai, NodeIdx to, Cap cap, BoundaryKey key)
{
    ArcIdx next_in_boundary = register_boundary_arc(ai, key);
    Arc& a = arcs[ai];
    a.head = to;
    a.next = next_in_boundary;
    a.r_cap = cap;
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::build_half_edge(
    ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap)
{
    Arc& a = arcs[ai];
    a.head = to;
    a.r_cap = cap;
    a.next = nodes[from].first;
    nodes[from].first = ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::add_outgoing(NodeIdx from, ArcIdx ai)
{
    arcs[ai].next = nodes[from].first;
    nodes[from].first = ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline bool ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::should_activate(NodeIdx i, NodeIdx j)
{
    Node &ni = nodes[i];
    Node &nj = nodes[j];
    // If one of the nodes were previously visited, but the other wasn't, 
    // or they are different (source/sink) and have both previously been visited.
    return (ni.parent == INVALID_ARC && nj.parent != INVALID_ARC) ||
        (ni.parent != INVALID_ARC && nj.parent == INVALID_ARC) ||
        (ni.parent != INVALID_ARC && nj.parent != INVALID_ARC && ni.is_sink != nj.is_sink);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline BoundaryKey ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::block_key(
    BlockIdx i, BlockIdx j) const noexcept
{
    constexpr BoundaryKey shift = sizeof(BlockIdx) * 8;
    if (i < j) {
        return static_cast<BoundaryKey>(i) | (static_cast<BoundaryKey>(j) << shift);
    } else {
        return static_cast<BoundaryKey>(j) | (static_cast<BoundaryKey>(i) << shift);
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::pair<BlockIdx, BlockIdx> ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::blocks_from_key(
    BoundaryKey key) const noexcept
{
    static_assert(sizeof(std::pair<BlockIdx, BlockIdx>) == sizeof(BoundaryKey),
        "Pair of BlockIdx does not match size of BoundaryKey");
    // Since std::pair just stores two BlockIdx fields adjacent in memory, we can just reinterpret the key
    return *reinterpret_cast<std::pair<BlockIdx, BlockIdx> *>(&key);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::tuple<
    std::list<typename ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::BoundarySegment>, BlockIdx, BlockIdx, bool>
ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::next_boundary_segment_set()
{
    // NOTE: We assume the global lock is grabbed at this point so no other threads are scanning
    std::list<BoundarySegment> out;
    BlockIdx bi = 0, bj = 0;
    bool submodular_boundary = true;

    // Scan until we find a boundary segment with unlocked blocks
    auto iter = boundary_segments.begin();
    for (; iter != boundary_segments.end(); ++iter) {
        const auto& bs = *iter;
        if (!blocks[block_idxs[bs.i]].locked && !blocks[block_idxs[bs.j]].locked) {
            // Found boundary between two unlocked blocks
            bi = block_idxs[bs.i];
            bj = block_idxs[bs.j];
            break;
        }
    }
    // If we are not at the end, move all relevant boundary segments to output
    // Note that iter starts at the found boundary segment or at the end
    while (iter != boundary_segments.end()) {
        const auto& bs = *iter;
        if ((block_idxs[bs.i] == bi && block_idxs[bs.j] == bj) || 
            (block_idxs[bs.i] == bj && block_idxs[bs.j] == bi)) {
            // Boundary connectes blocks
            out.push_back(bs);
            if (!bs.all_submodular) {
                submodular_boundary = false;
            }
            auto to_erase = iter;
            ++iter;
            boundary_segments.erase(to_erase);
        } else {
            ++iter;
        }
    }
    return std::make_tuple(out, bi, bj, submodular_boundary);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::unite_blocks(BlockIdx i, BlockIdx j)
{
    auto& block = blocks[i];
    const auto& other = blocks[j];
    assert(block.stage == other.stage);
    block.time = std::max(block.time, other.time);
    block.flow += other.flow;
    block.all_submodular = block.all_submodular && other.all_submodular;
    
    // Merge linked lists
    assert(next_block_node[block.last_node] == INVALID_NODE);
    assert(next_block_arc[block.last_arc] == INVALID_ARC);
    next_block_node[block.last_node] = other.first_node;
    block.last_node = other.last_node;
    next_block_arc[block.last_arc] = other.first_arc;
    block.last_arc = other.last_arc;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::transform_to_second_stage(BlockIdx block_idx)
{
    // We assume that the node and arc array have already been expanded
    const size_t num_nodes = nodes.size() - node_shift;
    const size_t num_arcs = arcs.size() - arc_shift;

    auto& block = blocks[block_idx];

    block.time++;
    block.flow *= 2;

    // Make dual copy of primal nodes
    for (NodeIdx i = block.first_node; i != INVALID_NODE; i = next_block_node[i]) {
        Node& primal = nodes[i];
        Node& dual = nodes[dual_node(i)];

        dual.tr_cap = -primal.tr_cap;
        dual.is_sink = !primal.is_sink;
        dual.timestamp = primal.timestamp;
        dual.dist = primal.dist;
        if (primal.parent == INVALID_ARC || primal.parent == TERMINAL_ARC) {
            dual.parent = primal.parent;
        } else {
            dual.parent = dual_arc(sister_idx(primal.parent));
        }
    }

    // Make dual copy of primal arcs
    for (ArcIdx ai = block.first_arc; ai != INVALID_ARC; ai = next_block_arc[ai]) {
        const Arc& arc = arcs[ai];
        const Arc& sister_arc = sister(ai);
        assert(sister_idx(ai) == ai + 1);
        assert(sister_arc.head != INVALID_NODE);

        if (is_primal_node(arc.head)) {
            // Normal arc added from a submodular energy term
            NodeIdx di = dual_node(sister_arc.head);
            NodeIdx dj = dual_node(arc.head);

            // Add reversed copy in dual graph
            build_half_edge(dual_arc(ai), dj, di, arc.r_cap);
            build_half_edge(dual_arc(ai) + 1, di, dj, sister_arc.r_cap);
        } else {
            // Primal-to-dual arc added from non-submodular energy term
            NodeIdx pi = sister_arc.head;
            NodeIdx di = dual_node(pi);
            NodeIdx dj = arc.head;
            NodeIdx pj = primal_node(dj);

            // Add arcs as outgoing to nodes
            add_outgoing(pi, ai);
            add_outgoing(dj, sister_idx(ai));

            // Add mirror primal-to-dual copy
            build_half_edge(dual_arc(ai), pj, di, arc.r_cap);
            build_half_edge(dual_arc(ai) + 1, di, pj, sister_arc.r_cap);

            // Mark nodes as active
            block.make_active(pi);
            block.make_active(di);
            block.make_active(pj);
            block.make_active(dj);
        }
    }

    block.stage = 1;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::register_block_nodes(
    NodeIdx first, NodeIdx last, BlockIdx block)
{
    auto& b = blocks[block];
    if (b.last_node == INVALID_NODE) {
        // First registered node
        b.first_node = first;
    } else {
        next_block_node[b.last_node] = first;
    }
    b.last_node = last;
    std::iota(next_block_node.begin() + first, next_block_node.begin() + last, first + 1);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::register_block_arc(ArcIdx i, BlockIdx block)
{
    auto& b = blocks[block];
    if (b.last_arc == INVALID_ARC) {
        // First registered arc
        b.first_arc = i;
    } else {
        next_block_arc[b.last_arc] = i;
    }
    b.last_arc = i;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::register_boundary_arc(ArcIdx a, BoundaryKey key)
{
    auto res = boundary_arcs.insert({ key, a });
    if (!res.second) {
        // Value was not inserted since a value already exists
        std::swap(res.first->second, a);
        return a;
    } else {
        // Value was inserted so it is the first element and there is no next arc
        return INVALID_ARC;
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::mirror_node(NodeIdx i) const noexcept
{
    return is_primal_node(i) ? dual_node(i) : primal_node(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::mirror_arc(ArcIdx i) const noexcept
{
    return is_primal_arc(i) ? dual_arc(i) : primal_arc(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Flow ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::maxflow()
{
    NodeIdx crnt_node = INVALID_NODE;

    // main loop
    while (true) {
        NodeIdx i = crnt_node;

        // Check if we are already exploring a valid active node
        if (i != INVALID_NODE) {
            Node& n = nodes[i];
            n.next_active = INVALID_NODE;
            if (n.parent == INVALID_ARC) {
                // Active node was not valid so don't explore after all
                i = INVALID_NODE;
            }
        }

        // If we are not already exploring a node try to get a new one
        if (i == INVALID_NODE) {
            i = next_active();
            if (i == INVALID_NODE) {
                // No more nodes to explore so we are done
                break;
            }
        }

        // At this point i must point to a valid active node
        ArcIdx source_sink_connector = grow_search_tree(i);

#ifndef SHRDR_NO_OVERFLOW_CHECKS
        // Check for overflow in time variable.
        if (time == std::numeric_limits<Time>::max()) {
            throw std::overflow_error("Overflow in 'time' variable. Please increase capacity of Time type.");
        }
#endif
        time++;

        if (source_sink_connector != INVALID_ARC) {
            // Growth was aborted because we found a node from the other tree
            nodes[i].next_active = i; // Mark as active
            crnt_node = i;

            augment(source_sink_connector);

            std::deque<NodeIdx> crnt_orphan_nodes = std::move(orphan_nodes); // Snapshot of current ophans
            for (NodeIdx orphan : crnt_orphan_nodes) {
                process_orphan(orphan);
                // If any additional orphans were added during processing, we process them immediately
                // This leads to a significant decrease of the overall runtime
                while (!orphan_nodes.empty()) {
                    NodeIdx o = orphan_nodes.front();
                    orphan_nodes.pop_front();
                    process_orphan(o);
                }
            }
        } else {
            crnt_node = INVALID_NODE;
        }

    }

    return flow;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::make_active(NodeIdx i)
{
    if (nodes[i].next_active == INVALID_NODE) {
        // It's not in the active list yet
        if (last_active != INVALID_NODE) {
            nodes[last_active].next_active = i;
        } else {
            first_active = i;
        }
        last_active = i;
        nodes[i].next_active = i;
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::make_front_orphan(NodeIdx i)
{
    nodes[i].parent = ORPHAN_ARC;
    orphan_nodes.push_front(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::make_back_orphan(NodeIdx i)
{
    nodes[i].parent = ORPHAN_ARC;
    orphan_nodes.push_back(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::next_active()
{
    NodeIdx i;
    // Pop nodes from the active list until we find a valid one or run out of nodes
    for (i = first_active; i != INVALID_NODE; i = first_active) {
        // Pop node from active list
        Node& n = nodes[i];
        if (n.next_active == i) {
            // This is the last node in the active list so "clear" it
            first_active = INVALID_NODE;
            last_active = INVALID_NODE;
        } else {
            first_active = n.next_active;
        }
        n.next_active = INVALID_NODE; // Mark as not active

        if (n.parent != INVALID_ARC) {
            // Valid active node found
            break;
        }
    }
    return i;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::augment(ArcIdx middle_idx)
{
    Arc& middle = arcs[middle_idx];
    Arc& middle_sister = sister(middle_idx);
    // Step 1: Find bottleneck capacity
    Cap bottleneck = middle.r_cap;
    bottleneck = std::min<Cap>(bottleneck, tree_bottleneck(middle_sister.head, true));
    bottleneck = std::min<Cap>(bottleneck, tree_bottleneck(middle.head, false));

    // Step 2: Augment along source and sink tree
    middle_sister.r_cap += bottleneck;
    middle.r_cap -= bottleneck;
    augment_tree(middle_sister.head, bottleneck, true);
    augment_tree(middle.head, bottleneck, false);

    // Step 3: Add bottleneck to overall flow
    flow += bottleneck;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Cap ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::tree_bottleneck(
    NodeIdx start, bool source_tree) const
{
    NodeIdx i = start;
    Cap bottleneck = std::numeric_limits<Cap>::max();
    while (true) {
        ArcIdx a = nodes[i].parent;
        if (a == TERMINAL_ARC) {
            break;
        }
        Cap r_cap = source_tree ? sister(a).r_cap : arcs[a].r_cap;
        bottleneck = std::min<Cap>(bottleneck, r_cap);
        i = arcs[a].head;
    }
    const Cap tr_cap = nodes[i].tr_cap;
    return std::min<Cap>(bottleneck, source_tree ? tr_cap : -tr_cap);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::augment_tree(
    NodeIdx start, Cap bottleneck, bool source_tree)
{
    NodeIdx i = start;
    while (true) {
        ArcIdx ai = nodes[i].parent;
        if (ai == TERMINAL_ARC) {
            break;
        }
        Arc& a = source_tree ? arcs[ai] : sister(ai);
        Arc& b = source_tree ? sister(ai) : arcs[ai];
        a.r_cap += bottleneck;
        b.r_cap -= bottleneck;
        if (b.r_cap == 0) {
            make_front_orphan(i);
        }
        i = arcs[ai].head;
    }
    nodes[i].tr_cap += source_tree ? -bottleneck : bottleneck;
    if (nodes[i].tr_cap == 0) {
        make_front_orphan(i);
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::grow_search_tree(NodeIdx start)
{
    return nodes[start].is_sink ? grow_search_tree_impl<false>(start) : grow_search_tree_impl<true>(start);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
template<bool source>
inline ArcIdx ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::grow_search_tree_impl(NodeIdx start_idx)
{
    const Node& start = nodes[start_idx];
    ArcIdx ai;
    // Add neighbor nodes search tree until we find a node from the other search tree or run out of neighbors
    for (ai = start.first; ai != INVALID_ARC; ai = arcs[ai].next) {
        if (sister_or_arc(ai, !source).r_cap) {
            Node& n = head_node(ai);
            if (n.parent == INVALID_ARC) {
                // This node is not yet in a tree so claim it for this one
                n.is_sink = !source;
                n.parent = sister_idx(ai);
                n.timestamp = start.timestamp;
                n.dist = start.dist + 1;
                make_active(arcs[ai].head);
            } else if (n.is_sink == source) {
                // Found a node from the other search tree so abort
                if (!source) {
                    // If we are growing the sink tree we instead return the sister arc
                    ai = sister_idx(ai);
                }
                break;
            } else if (n.timestamp <= start.timestamp && n.dist > start.dist) {
                // Heuristic: trying to make the distance from j to the source/sink shorter
                n.parent = sister_idx(ai);
                n.timestamp = n.timestamp;
                n.dist = start.dist + 1;
            }
        }
    }
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::process_orphan(NodeIdx i)
{
    if (nodes[i].is_sink) {
        process_orphan_impl<false>(i);
    } else {
        process_orphan_impl<true>(i);
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
template<bool source>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::QpboBlock::process_orphan_impl(NodeIdx i)
{
    Node &n = nodes[i];
    static const int32_t INF_DIST = std::numeric_limits<int32_t>::max();
    int32_t min_d = INF_DIST;
    ArcIdx min_a0 = INVALID_ARC;
    // Try to find a new parent
    for (ArcIdx a0 = n.first; a0 != INVALID_ARC; a0 = arcs[a0].next) {
        if (sister_or_arc(a0, source).r_cap) {
            NodeIdx j = arcs[a0].head;
            ArcIdx a = nodes[j].parent;
            if (nodes[j].is_sink != source && a != INVALID_ARC) {
                // Check origin of m
                int32_t d = 0;
                while (true) {
                    Node &m = nodes[j];
                    if (m.timestamp == time) {
                        d += m.dist;
                        break;
                    }
                    a = m.parent;
                    d++;
                    if (a == TERMINAL_ARC) {
                        m.timestamp = time;
                        m.dist = 1;
                        break;
                    }
                    if (a == ORPHAN_ARC) {
                        d = INF_DIST; // infinite distance
                        break;
                    }
                    j = arcs[a].head;
                }
                if (d < INF_DIST) {
                    // m originates from the source
                    if (d < min_d) {
                        min_a0 = a0;
                        min_d = d;
                    }
                    // Set marks along the path
                    j = arcs[a0].head;
                    while (nodes[j].timestamp != time) {
                        Node &m = nodes[j];
                        m.timestamp = time;
                        m.dist = d--;
                        j = arcs[m.parent].head;
                    }
                }
            }
        }
    }
    n.parent = min_a0;
    if (min_a0 != INVALID_ARC) {
        n.timestamp = time;
        n.dist = min_d + 1;
    } else {
        // No parent was found so process neighbors
        for (ArcIdx a0 = n.first; a0 != INVALID_ARC; a0 = arcs[a0].next) {
            NodeIdx j = arcs[a0].head;
            Node &m = nodes[j];
            if (m.is_sink != source && m.parent != INVALID_ARC) {
                if (sister_or_arc(a0, source).r_cap) {
                    make_active(j);
                }
                ArcIdx pa = m.parent;
                if (pa != TERMINAL_ARC && pa != ORPHAN_ARC && arcs[pa].head == i) {
                    make_back_orphan(j);
                }
            }
        }
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void ParallelQpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_weak_persistencies()
{
    // If we're in stage 0, don't do anything
    if (stage == 0) {
        return;
    }

    // Initialize search
    for (NodeIdx pi = 0; pi < get_primal_node_num(); pi++) {
        NodeIdx di = dual_node(pi);
        Node& p = nodes[pi];
        Node& d = nodes[di];

        if (p.label != UNKNOWN) {
            p.dfs_parent = pi;
            p.region = 0;
            d.dfs_parent = di;
            d.region = 0;
        } else {
            p.dfs_parent = INVALID_NODE;
            p.region = -1;
            d.dfs_parent = INVALID_NODE;
            d.region = -1;
        }
    }

    NodeIdx stack = INVALID_NODE;

    // First depth first search
    for (NodeIdx i = 0; i < get_total_node_num(); i++)
    {
        Node& n = nodes[i];

        if (n.dfs_parent != INVALID_NODE) {
            continue;
        }

        n.dfs_parent = i;
        n.dfs_current = n.first;

        while (true) {
            if (nodes[i].dfs_current == INVALID_ARC) {
                nodes[i].next_active = stack;
                stack = i;

                if (nodes[i].dfs_parent == i) {
                    break;
                }

                i = nodes[i].dfs_parent;
                nodes[i].dfs_current = arcs[nodes[i].dfs_current].next;
                continue;
            }

            Arc& a = arcs[nodes[i].dfs_current];
            NodeIdx j = a.head;
            if (a.r_cap == 0 || nodes[j].dfs_parent != INVALID_NODE) {
                nodes[i].dfs_current = a.next;
                continue;
            }

            nodes[j].dfs_parent = i;
            i = j;
            nodes[i].dfs_current = nodes[i].first;
        }
    }

    // Second depth first search
    int32_t component = 0;
    while (stack != INVALID_NODE) {

        NodeIdx i = stack;
        Node& n = nodes[i];
        stack = n.next_active;

        if (n.region > 0) {
            continue;
        }

        component++;
        n.region = component;
        n.dfs_parent = i;
        n.dfs_current = n.first;

        while (true) {
            if (nodes[i].dfs_current == INVALID_ARC) {
                if (nodes[i].dfs_parent == i) {
                    break;
                }
                i = nodes[i].dfs_parent;
                nodes[i].dfs_current = arcs[nodes[i].dfs_current].next;
                continue;
            }

            ArcIdx a_idx = nodes[i].dfs_current;
            Arc& a = arcs[a_idx];
            NodeIdx j = a.head;
            if (sister(a_idx).r_cap == 0 || nodes[j].region >= 0) {
                nodes[i].dfs_current = a.next;
                continue;
            }

            nodes[j].dfs_parent = i;
            i = j;
            nodes[i].dfs_current = nodes[i].first;
            nodes[i].region = component;
        }
    }

    // Assign labels
    for (NodeIdx pi = 0; pi < get_primal_node_num(); pi++) {
        Node& p = nodes[pi];

        if (p.label == UNKNOWN) {
            NodeIdx di = dual_node(pi);
            Node& d = nodes[di];

            assert(p.region > 0);

            if (p.region > d.region) {
                p.label = SOURCE;
                p.region = 0;
            }
            else if (p.region < d.region) {
                p.label = SINK;
                p.region = 0;
            }
        }
        else {
            assert(p.region == 0);
        }
    }
}

} // namespace shrdr

#endif // SHRDR_PARALLEL_QPBO_H__