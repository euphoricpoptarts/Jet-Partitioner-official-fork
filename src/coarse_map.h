#pragma once

// a mapping from a fine grapher to a coarser graph
template <class vtx_vt>
struct coarse_map {
    using ordinal_t = typename vtx_vt::non_const_value_type;

    ordinal_t coarse_vtx;
    vtx_vt map;
};