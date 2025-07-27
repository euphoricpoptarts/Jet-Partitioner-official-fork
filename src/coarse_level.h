#pragma once
#include <Kokkos_Core.hpp>
#include "coarse_map.h"

// contains matrix and vertex weights/penalties corresponding to current level
// interp matrix maps previous level to this level
template <class matrix_t>
struct coarse_level {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using vtx_vt = Kokkos::View<ordinal_t*, Device>;
    using coarse_map_t = coarse_map<vtx_vt>;

    matrix_t mtx;
    wgt_vt vtx_w;
    wgt_vt wdeg;
    coarse_map_t interp_mtx;
    int level;
    bool uniform_weights = false;
};