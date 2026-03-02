// Copyright 2026 Michael S. Gilbert II
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "memory_store.hpp"
#include "cluster_data.hpp"
#include "weighted_graph.h"
#include "core_types.h"
#include "coarse_level.h"
#include <list>

namespace jet_community {
namespace clustering_methods {
    // define internal types
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using vtx_vt = typename Kokkos::View<ordinal_t*, Device>;
    using mem_t = memory_store<matrix_t, int>;
    using wg_t = weighted_graph;
    using rfd_t = cluster_data;
    using coarse_level_t = coarse_level<matrix_t>;

    template <bool plus, bool improve>
    std::list<coarse_level_t> leiden_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt input, const ordinal_t upper_bound);

    template <bool constrained>
    std::list<coarse_level_t> louvain_part(mem_t& mem, wg_t top, rfd_t& rfd, vtx_vt constraint, const ordinal_t upper_bound);

}
}