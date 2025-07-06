#pragma once
#include <type_traits>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_CrsMatrix.hpp"

namespace jet_community {

template <class crsMat, typename part_t>
struct memory_store {

    //helper for getting gain_t
    template<typename T>
    struct type_identity {
        typedef T type;
    };

    // define internal types
    using matrix_t = crsMat;
    using Device = typename matrix_t::device_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using edge_offset_t = typename matrix_t::size_type;
    using scalar_t = typename matrix_t::value_type;
    // need some trickery because make_signed is undefined for floating point types
    using gain_t = typename std::conditional_t<std::is_signed_v<scalar_t>, type_identity<scalar_t>, std::make_signed<scalar_t>>::type;
    using vtx_view_t = Kokkos::View<ordinal_t*, Device>;
    using edge_view_t = Kokkos::View<edge_offset_t*, Device>;
    using gain_vt = Kokkos::View<gain_t*, Device>;
    using vtx_pin_st = Kokkos::View<ordinal_t, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_vt = Kokkos::View<gain_t*, Kokkos::SharedHostPinnedSpace>;
    using gain_pin_st = Kokkos::View<gain_t, Kokkos::SharedHostPinnedSpace>;
    using part_vt = Kokkos::View<part_t*, Device>;
    using obj_vt = Kokkos::View<float*, Device>;

    // vertex-part connectivity datastructure
    struct conn_data {
        edge_view_t conn_offsets;
        gain_vt conn_vals;
        part_vt conn_entries;
        // matrix wrapper of above views
        matrix_t c_graph;
        part_vt conn_table_sizes;
        bool init = false;

        conn_data(const matrix_t largest){
            ordinal_t n = largest.numRows();
            conn_vals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("conn vals"), largest.nnz());
            conn_entries = part_vt(Kokkos::ViewAllocateWithoutInitializing("conn entries"), largest.nnz());
            conn_offsets = edge_view_t(Kokkos::ViewAllocateWithoutInitializing("conn offsets"), n + 1);
            conn_table_sizes = part_vt(Kokkos::ViewAllocateWithoutInitializing("conn table size"), n);
        }
    };

    // this struct contains memory for persistent state across lp iterations
    struct persistent {
        obj_vt gain_persistent;
        gain_vt pvals;
        part_vt part, dest_part;
        vtx_view_t order1, order2;
        ordinal_t offset_mid, offset_large;

        persistent(const ordinal_t n){
            gain_persistent = obj_vt(Kokkos::ViewAllocateWithoutInitializing("gain persistent"), n);
            pvals = gain_vt(Kokkos::ViewAllocateWithoutInitializing("p vals"), n);
            part = part_vt(Kokkos::ViewAllocateWithoutInitializing("part scratch"), n);
            dest_part = part_vt(Kokkos::ViewAllocateWithoutInitializing("destination scratch"), n);
            order1 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 1"), n);
            order2 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx ordering 2"), n);
        }

        persistent(const persistent& source, const ordinal_t n){
            gain_persistent = Kokkos::subview(source.gain_persistent, std::make_pair(static_cast<ordinal_t>(0), n));
            pvals = Kokkos::subview(source.pvals, std::make_pair(static_cast<ordinal_t>(0), n));
            part = Kokkos::subview(source.part, std::make_pair(static_cast<ordinal_t>(0), n));
            dest_part = Kokkos::subview(source.dest_part, std::make_pair(static_cast<ordinal_t>(0), n));
            order1 = source.order1;
            order2 = source.order2;
        }
    };

    // this struct contains memory which does not require initialization between lp iterations
    struct scratch {
        vtx_view_t vtx1, vtx2;
        vtx_view_t zeros1;
        vtx_pin_st scan_host, pin_host;
        gain_pin_st cut_change1, cut_change2;
        gain_pin_vt reduce_locs;

        scratch(const ordinal_t n) {
            vtx1 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 1"), n);
            vtx2 = vtx_view_t(Kokkos::ViewAllocateWithoutInitializing("vtx scratch 2"), n);
            zeros1 = vtx_view_t("zeros 1", n);
            scan_host = vtx_pin_st("scan host");
            pin_host = vtx_pin_st("pin host");
            reduce_locs = gain_pin_vt("reduce to here", 2);
            cut_change1 = Kokkos::subview(reduce_locs, 0);
            cut_change2 = Kokkos::subview(reduce_locs, 1);
        }
    };

    conn_data cd_mem;
    persistent p_mem;
    scratch s_mem;

    memory_store(const matrix_t largest) :
        cd_mem(largest),
        p_mem(largest.numRows()), 
        s_mem(largest.numRows()) {}

    memory_store(memory_store& source, const matrix_t g) :
        cd_mem(source.cd_mem),
        p_mem(source.p_mem, g.numRows()),
        s_mem(source.s_mem) {}
};
}