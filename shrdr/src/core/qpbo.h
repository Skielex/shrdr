#ifndef SHRDR_QPBO_H__
#define SHRDR_QPBO_H__

#include <list>
#include <unordered_map>
#include <deque>
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <cinttypes>
#include <cassert>
#include <algorithm>
#include <type_traits>
#include <vector>

#include "util.h"

namespace shrdr {

using Time = uint32_t;
using Dist = uint16_t;

template <
    class Cap,
    class Flow = typename std::conditional<std::is_floating_point<Cap>::value, double, int64_t>::type,
    class ArcIdx = uint32_t,
    class NodeIdx = uint32_t
>
class Qpbo {
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

    Qpbo(size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular);

    NodeIdx add_node(size_t num = 1);

    void add_unary_term(NodeIdx i, Cap e0, Cap e1);

    void add_pairwise_term(NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11);

    NodeLabel get_label(NodeIdx i) const;
    NodeLabel what_segment(NodeIdx i, NodeLabel default_segment = SOURCE) const;

    std::pair<Cap, Cap> get_twice_unary_term(NodeIdx i) const;
    std::tuple<Cap, Cap, Cap, Cap> get_twice_pairwise_term(ArcIdx term) const;

    std::pair<NodeIdx, NodeIdx> get_arc_ends(ArcIdx ai) const;

    Flow compute_twice_energy() const;
    inline Cap get_flow() const noexcept { return flow; };

    NodeIdx get_primal_node_num() const noexcept { return nodes.size() - (stage > 0) * node_shift; };
    NodeIdx get_total_node_num() const noexcept { return nodes.size(); };
    ArcIdx get_primal_arc_num() const noexcept { return arcs.size() - (stage > 0) * arc_shift; };
    ArcIdx get_total_arc_num() const noexcept { return arcs.size(); };

    void solve();
    
    void compute_weak_persistencies();

private:
    std::vector<Node> nodes;
    std::vector<Arc> arcs;

    Flow zero_energy;
    Flow flow;

    Time time;

    NodeIdx first_active, last_active;
    std::deque<NodeIdx> orphan_nodes;

    NodeIdx node_shift;
    ArcIdx arc_shift;

    bool all_submodular;
    int stage;

    void init_maxflow();
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

    /** Build primal-dual graph construction and add non-submodular edges */
    void transform_to_second_stage();

    std::tuple<Cap, Cap, Cap, Cap> compute_normal_form_weights(
        Cap e00, Cap e01, Cap e10, Cap e11) const noexcept;

    ArcIdx add_half_edge(NodeIdx from, NodeIdx to, Cap cap);
    ArcIdx add_half_edge(NodeIdx to, Cap cap);
    void build_half_edge(ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap);

    void add_outgoing(NodeIdx from, ArcIdx ai);

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
const NodeIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::INVALID_NODE;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::INVALID_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::TERMINAL_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
const ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::ORPHAN_ARC;

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Qpbo<Cap, Flow, ArcIdx, NodeIdx>::Qpbo(
    size_t expected_nodes, size_t expected_pairwise_terms, bool expect_nonsubmodular) :
    nodes(),
    arcs(),
    zero_energy(0),
    flow(0),
    time(0),
    first_active(INVALID_NODE),
    last_active(INVALID_NODE),
    orphan_nodes(),
    node_shift(expected_nodes),
    arc_shift(0),
    all_submodular(true),
    stage(0)
{
    nodes.reserve((expect_nonsubmodular ? 2 : 1) * expected_nodes);
    arcs.reserve((expect_nonsubmodular ? 4 : 2) * expected_pairwise_terms);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_node(size_t num)
{
    NodeIdx crnt = nodes.size();

#ifndef SHRDR_NO_OVERFLOW_CHECKS
    if (crnt > std::numeric_limits<NodeIdx>::max() - num) {
        throw std::overflow_error("Node count exceeds capacity of index type. "
            "Please increase capacity of NodeIdx type.");
    }
#endif

    if (!all_submodular && crnt + num > node_shift) {
        // Since non-submodular terms have been added, we first need to to update all head indices for all
        // arcs between the primal and dual graph
        node_shift += num;
        for (auto& a : arcs) {
            if (a.head >= crnt) {
                a.head += num;
            }
        }
    }

    nodes.resize(crnt + num);
    return crnt;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_unary_term(NodeIdx i, Cap e0, Cap e1)
{
    nodes[i].tr_cap += e1 - e0;
    zero_energy += e0;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_pairwise_term(
    NodeIdx i, NodeIdx j, Cap e00, Cap e01, Cap e10, Cap e11)
{
#ifndef SHRDR_NO_OVERFLOW_CHECKS
    if (arcs.size() > std::numeric_limits<ArcIdx>::max() - 2) {
        throw std::overflow_error("Arc count exceeds capacity of index type. "
            "Please increase capacity of ArcIdx type.");
    }
#endif
    Cap ci, cj, cij, cji;
    if (e00 + e11 <= e01 + e10) {
        // Cap is submodular
        std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e00, e01, e10, e11);

        add_half_edge(i, j, cij);
        add_half_edge(j, i, cji);
    } else {
        // Cap is *not* submodular
        all_submodular = false;
        // Note that energy coefs. are switched!
        std::tie(ci, cj, cij, cji) = compute_normal_form_weights(e01, e00, e11, e10);

        // We only set head and r_cap here and update next when we move to the second stage
        add_half_edge(dual_node(j), cij);
        add_half_edge(i, cji);
    }

    // Terminal arcs
    nodes[i].tr_cap += ci;
    nodes[j].tr_cap += cj;

    arc_shift += 2;
    zero_energy += e00;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeLabel Qpbo<Cap, Flow, ArcIdx, NodeIdx>::get_label(NodeIdx i) const
{
    assert(0 <= i && is_primal_node(i));
    return nodes[i].label;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeLabel Qpbo<Cap, Flow, ArcIdx, NodeIdx>::what_segment(NodeIdx i, NodeLabel default_segment) const
{
    assert(0 <= i && i < nodes.size());
    if (nodes[i].parent != INVALID_ARC) {
        return (nodes[i].is_sink) ? SINK : SOURCE;
    } else {
        return default_segment;
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::pair<Cap, Cap> Qpbo<Cap, Flow, ArcIdx, NodeIdx>::get_twice_unary_term(NodeIdx i) const
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
inline std::tuple<Cap, Cap, Cap, Cap> Qpbo<Cap, Flow, ArcIdx, NodeIdx>::get_twice_pairwise_term(ArcIdx term) const
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
inline std::pair<NodeIdx, NodeIdx> Qpbo<Cap, Flow, ArcIdx, NodeIdx>::get_arc_ends(ArcIdx ai) const
{
    assert(0 <= ai && ai <= arcs.size());
    return std::make_pair(arcs[ai].head, sister(ai).head);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Flow Qpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_twice_energy() const
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::solve()
{
    init_maxflow();
    maxflow();

    if (!all_submodular) {
        transform_to_second_stage();
        maxflow();
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::init_maxflow()
{
    first_active = INVALID_NODE;
    last_active = INVALID_NODE;
    orphan_nodes.clear();
    time = 0;

    for (size_t i = 0; i < nodes.size(); ++i) {
        Node& n = nodes[i];
        n.next_active = INVALID_NODE;
        n.timestamp = time;
        if (n.tr_cap != 0) {
            // n is connected to the source or sink
            n.is_sink = n.tr_cap < 0; // negative capacity goes to sink
            n.parent = TERMINAL_ARC;
            n.dist = 1;
            make_active(i);
        } else {
            n.parent = INVALID_ARC;
        }
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Flow Qpbo<Cap, Flow, ArcIdx, NodeIdx>::maxflow()
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::make_active(NodeIdx i)
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::make_front_orphan(NodeIdx i)
{
    nodes[i].parent = ORPHAN_ARC;
    orphan_nodes.push_front(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::make_back_orphan(NodeIdx i)
{
    nodes[i].parent = ORPHAN_ARC;
    orphan_nodes.push_back(i);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline NodeIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::next_active()
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::augment(ArcIdx middle_idx)
{
    Arc& middle = arcs[middle_idx];
    Arc& middle_sister = sister(middle_idx);
    // Step 1: Find bottleneck capacity
    Cap bottleneck = middle.r_cap;
    bottleneck = std::min<Cap>(bottleneck, tree_bottleneck(middle_sister.head, true));
    bottleneck = std::min<Cap>(bottleneck, tree_bottleneck(middle.head, false));

    // Step  2: Augment along source and sink tree
    middle_sister.r_cap += bottleneck;
    middle.r_cap -= bottleneck;
    augment_tree(middle_sister.head, bottleneck, true);
    augment_tree(middle.head, bottleneck, false);

    // Step 3: Add bottleneck to overall flow
    flow += bottleneck;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline Cap Qpbo<Cap, Flow, ArcIdx, NodeIdx>::tree_bottleneck(NodeIdx start, bool source_tree) const
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::augment_tree(NodeIdx start, Cap bottleneck, bool source_tree)
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
inline ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::grow_search_tree(NodeIdx start)
{
    return nodes[start].is_sink ? grow_search_tree_impl<false>(start) : grow_search_tree_impl<true>(start);
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
template<bool source>
inline ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::grow_search_tree_impl(NodeIdx start_idx)
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::process_orphan(NodeIdx i)
{
    if (nodes[i].is_sink) {
        process_orphan_impl<false>(i);
    } else {
        process_orphan_impl<true>(i);
    }
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
template<bool source>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::process_orphan_impl(NodeIdx i)
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
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::transform_to_second_stage()
{
    const size_t num_nodes = nodes.size();
    const size_t num_arcs = arcs.size();
    // If num_nodes < dual_shift, the primal nodes are padded with dummy nodes
    nodes.resize(node_shift + num_nodes);
    arcs.resize(2 * arcs.size());

    time++;
    flow *= 2;

    // Make dual copy of primal nodes
    for (int i = 0; i < num_nodes; ++i) {
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
    for (int ai = 0; ai < num_arcs; ai += 2) {
        const Arc& arc = arcs[ai];
        const Arc& sister_arc = sister(ai);
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

            // Add arc as outgoing node to nodes
            add_outgoing(pi, ai);
            add_outgoing(dj, sister_idx(ai));

            // Add mirror primal-to-dual copy
            build_half_edge(dual_arc(ai), pj, di, arc.r_cap);
            build_half_edge(dual_arc(ai) + 1, di, pj, sister_arc.r_cap);

            // Mark nodes as active
            make_active(pi);
            make_active(di);
            make_active(pj);
            make_active(dj);
        }
    }

    stage = 1;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline std::tuple<Cap, Cap, Cap, Cap> Qpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_normal_form_weights(
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
inline ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_half_edge(NodeIdx from, NodeIdx to, Cap cap)
{
    ArcIdx ai = arcs.size();
    arcs.emplace_back(to, nodes[from].first, cap);
    nodes[from].first = ai;
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline ArcIdx Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_half_edge(NodeIdx to, Cap cap)
{
    ArcIdx ai = arcs.size();
    arcs.emplace_back(to, INVALID_ARC, cap);
    return ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::build_half_edge(ArcIdx ai, NodeIdx from, NodeIdx to, Cap cap)
{
    Arc& a = arcs[ai];
    a.head = to;
    a.r_cap = cap;
    a.next = nodes[from].first;
    nodes[from].first = ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::add_outgoing(NodeIdx from, ArcIdx ai)
{
    arcs[ai].next = nodes[from].first;
    nodes[from].first = ai;
}

template<class Cap, class Flow, class ArcIdx, class NodeIdx>
inline void Qpbo<Cap, Flow, ArcIdx, NodeIdx>::compute_weak_persistencies()
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
            } else if (p.region < d.region) {
                p.label = SINK;
                p.region = 0;
            }
        } else {
            assert(p.region == 0);
        }
    }
}

} // namespace shrdr

#endif // SHRDR_QPBO_H__