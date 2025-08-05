#pragma once
#include <Kokkos_Core.hpp>

template <typename matrix_t>
struct cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using exec_space = typename matrix_t::execution_space;
    using policy_t = Kokkos::RangePolicy<exec_space>;

    // metadata that is preserved between levels in the clustering scheme
    wgt_vt total_deg;
    wgt_vt cluster_size;
    scalar_t g_deg = 0;
    scalar_t v_total = 0;
    scalar_t cut = 0;
    double obj = -1.0;
    ordinal_t label_count;

    // metadata needed between local move iterations
    scalar_t last_pval = 0;

    // objective scaling
    double penalty_scale = 2.0;

    static scalar_t sum(const wgt_vt wdeg){
        scalar_t result = 0;
        Kokkos::parallel_reduce("sum view", policy_t(0, wdeg.size()), KOKKOS_LAMBDA(const ordinal_t i, scalar_t& update){
            update += wdeg(i);
        }, result);
        return result;
    }

    cluster_data(const matrix_t g, const wgt_vt wdeg, const wgt_vt vtx_w, double _penalty_scale, bool uniform) {
        // this is not the true objective for a singleton clustering
        // but we should find a better one regardless so it doesn't matter
        obj = -1.0;
        total_deg = wgt_vt("total degree of clusters", g.numRows());
        cluster_size = wgt_vt("total size of clusters", g.numRows());
        if(uniform) g_deg = g.nnz();
        else g_deg = sum(g.values);
        v_total = g.numRows();
        cut = g_deg;
        label_count = g.numRows();
        Kokkos::deep_copy(exec_space(), total_deg, wdeg);
        Kokkos::deep_copy(exec_space(), cluster_size, vtx_w);
        penalty_scale = _penalty_scale;
    }

    void update(const matrix_t g, const wgt_vt wdeg, const wgt_vt vtx_w) {
        total_deg = wgt_vt("total deg", g.numRows());
        cluster_size = wgt_vt("cluster size", g.numRows());
        Kokkos::deep_copy(exec_space(), total_deg, wdeg);
        Kokkos::deep_copy(exec_space(), cluster_size, vtx_w);
        cut = sum(g.values);
        label_count = g.numRows();
        obj = objective();
    }

    void copy(const cluster_data& rhs){
        Kokkos::deep_copy(exec_space(), total_deg, rhs.total_deg);
        Kokkos::deep_copy(exec_space(), cluster_size, rhs.cluster_size);
        g_deg = rhs.g_deg;
        cut = rhs.cut;
        v_total = rhs.v_total;
        obj = rhs.obj;
        label_count = rhs.label_count;
        penalty_scale = rhs.penalty_scale;
    }

    cluster_data(const cluster_data& rhs){
        total_deg = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
        cluster_size = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.cluster_size.extent(0));
        copy(rhs);
    }

    double get_penalty_modifier() const {
        double modifier = penalty_scale * static_cast<float>(g_deg);
        double inv_vtot = 1.0 / static_cast<float>(v_total);
        modifier = modifier * inv_vtot * inv_vtot;
        return modifier;
    }

    double objective() const {
        double m = 0;
        // avoid implicit capture of "this"
        wgt_vt total = total_deg;
        Kokkos::parallel_reduce("sum of squares", policy_t(0, label_count), KOKKOS_LAMBDA(const ordinal_t l, double& update){
            double total_ratio = static_cast<double>(total(l));
            update -= total_ratio*total_ratio;
        }, m);
        double penalty_factor = 1.0 / static_cast<double>(v_total);
        penalty_factor = penalty_scale*penalty_factor*penalty_factor;
        m = m*penalty_factor;
        double inv_gdeg = 1.0 / static_cast<double>(g_deg);
        m += 1.0 - static_cast<double>(cut) * inv_gdeg;
        // std::cout << "Objective " << m << std::endl;
        return m;
    }
};