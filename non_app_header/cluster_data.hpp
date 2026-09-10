// Copyright 2026 Michael S. Gilbert II
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <Kokkos_Core.hpp>
#include <iostream>
#include <cstdint>
#include "core_types.h"
#include "weighted_graph.h"

struct cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using ordinal_t = typename matrix_t::ordinal_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;
    using exec_space = typename matrix_t::execution_space;
    using policy_t = Kokkos::RangePolicy<exec_space>;
    using wg_t = jet_community::weighted_graph;

    // metadata that is preserved between levels in the clustering scheme
    wgt_vt total_deg;
    wgt_vt total_wgt;
    bool reuse_deg_as_wgt;
    edge_offset_t uncut = 0;
    edge_offset_t top_nnz = 0;
    double obj = -1.0;
    ordinal_t label_count;
    int64_t save_square_sum = 0;

    // metadata needed between local move iterations
    edge_offset_t last_pval = 0;

    // objective scaling
    double lambda = 1.0;

    static edge_offset_t sum(const wgt_vt wdeg){
        edge_offset_t result = 0;
        Kokkos::parallel_reduce("sum view", policy_t(0, wdeg.size()), KOKKOS_LAMBDA(const ordinal_t i, edge_offset_t& update){
            update += wdeg(i);
        }, result);
        return result;
    }

    cluster_data(const wg_t wg, double _lambda) {
        matrix_t g = wg.mtx;
        reuse_deg_as_wgt = wg.reuse_w_as_pen;
        total_deg = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), g.numRows());
        if (reuse_deg_as_wgt) total_wgt = total_deg;
        else total_wgt = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total wgt of clusters"), g.numRows());
        Kokkos::deep_copy(exec_space(), total_deg, wg.v_pen);
        if(!reuse_deg_as_wgt) Kokkos::deep_copy(exec_space(), total_wgt, wg.vtx_w);
        top_nnz = g.nnz();
        uncut = 0;
        label_count = g.numRows();
        lambda = _lambda;
        update_objective();
    }

    void reset(const wg_t wg) {
        wgt_vt td_lhs = Kokkos::subview(total_deg, std::make_pair((ordinal_t)0, wg.mtx.numRows()));
        wgt_vt tw_lhs = Kokkos::subview(total_wgt, std::make_pair((ordinal_t)0, wg.mtx.numRows()));
        Kokkos::deep_copy(td_lhs, wg.v_pen);
        if(!reuse_deg_as_wgt) Kokkos::deep_copy(tw_lhs, wg.vtx_w);
        uncut = 0;
        label_count = wg.mtx.numRows();
        update_objective();
    }

    void copy(const cluster_data& rhs){
        reuse_deg_as_wgt = rhs.reuse_deg_as_wgt;
        wgt_vt td_lhs = Kokkos::subview(total_deg, std::make_pair((ordinal_t)0, rhs.label_count));
        wgt_vt td_rhs = Kokkos::subview(rhs.total_deg, std::make_pair((ordinal_t)0, rhs.label_count));
        Kokkos::deep_copy(exec_space(), td_lhs, td_rhs);
        if(!reuse_deg_as_wgt){
            wgt_vt tw_lhs = Kokkos::subview(total_wgt, std::make_pair((ordinal_t)0, rhs.label_count));
            wgt_vt tw_rhs = Kokkos::subview(rhs.total_wgt, std::make_pair((ordinal_t)0, rhs.label_count));
            Kokkos::deep_copy(exec_space(), tw_lhs, tw_rhs);
        }
        top_nnz = rhs.top_nnz;
        uncut = rhs.uncut;
        obj = rhs.obj;
        label_count = rhs.label_count;
        lambda = rhs.lambda;
        save_square_sum = rhs.save_square_sum;
    }

    cluster_data(const cluster_data& rhs){
        total_deg = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total degree of clusters"), rhs.total_deg.extent(0));
        if(rhs.reuse_deg_as_wgt) total_wgt = total_deg;
        else total_wgt = wgt_vt(Kokkos::ViewAllocateWithoutInitializing("total wgt of clusters"), rhs.total_wgt.extent(0));
        copy(rhs);
    }

    double get_penalty_modifier() const {
        return lambda;
    }

    void update_objective() {
        // avoid implicit capture of "this"
        wgt_vt total = total_deg;
        int64_t square_sum = 0;
        Kokkos::parallel_reduce("sum of squares", policy_t(0, label_count), KOKKOS_LAMBDA(const ordinal_t l, int64_t& update){
            int64_t c_size = total(l);
            update += c_size*c_size;
        }, square_sum);
        save_square_sum = square_sum;
        double m = uncut;
        m -= lambda * static_cast<double>(square_sum);
        obj = m;
    }

    // also updates objective
    void update_lambda(double _lambda){
        lambda = _lambda;
        double m = uncut;
        m -= lambda * static_cast<double>(save_square_sum);
        obj = m;
    }

    virtual void print(std::ostream& os) const {
        os << "Objective: " << obj;
    }

    // derived objectives need to apply scaling to obj
    virtual double get_objective() const {
        return obj;
    }

    friend std::ostream& operator<<(std::ostream& os, const cluster_data& cd) {
        os << "Cut: " << (cd.top_nnz - cd.uncut) / 2 << "; ";
        os << "Lambda: " << cd.lambda << "; ";
        cd.print(os);
        os << "; Labels: " << cd.label_count;
        return os;
    }

    virtual ~cluster_data(){}
};

// these derived classes manage the calculation of lambda and normalizing of the objective
struct modularity : public cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;

    double inv_gdeg = 0;

    modularity(const wg_t wg, double _penalty_scale) : cluster_data(wg, 1.0) {
        edge_offset_t g_deg = 0;
        if(wg.edge_uniform) g_deg = wg.mtx.nnz();
        else g_deg = cluster_data::sum(wg.v_pen);
        inv_gdeg = 1.0 / static_cast<double>(g_deg);
        double new_lambda = _penalty_scale * inv_gdeg;
        cluster_data::update_lambda(new_lambda);
    }

    virtual double get_objective() const override {
        return cluster_data::obj * inv_gdeg;
    }

    virtual void print(std::ostream& os) const override {
        os << "Modularity: " << get_objective();
    }

    virtual ~modularity(){}
};

struct constant_potts : public cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;

    ordinal_t v_total = 0;

    constant_potts(const wg_t wg, double _penalty_scale) : cluster_data(wg, 1.0) {
        v_total = wg.mtx.numRows();
        cluster_data::update_lambda(_penalty_scale);
    }

    virtual double get_objective() const override {
        return cluster_data::obj + cluster_data::lambda*v_total;
    }

    virtual void print(std::ostream& os) const override {
        os << "Constant-Potts: " << get_objective();
    }

    virtual ~constant_potts(){}
};

struct normalized_lcc : public cluster_data {
    using Device = typename matrix_t::device_type;
    using scalar_t = typename matrix_t::value_type;
    using wgt_vt = Kokkos::View<scalar_t*, Device>;

    edge_offset_t g_deg = 0;

    normalized_lcc(const wg_t wg, double _penalty_scale) : cluster_data(wg, 1.0) {
        if(wg.edge_uniform) g_deg = wg.mtx.nnz();
        else g_deg = cluster_data::sum(wg.mtx.values);
        uint64_t v_total = cluster_data::sum(wg.v_pen);
        double denom = static_cast<double>(v_total * v_total);
        double new_lambda = _penalty_scale * static_cast<double>(g_deg) / denom;
        cluster_data::update_lambda(new_lambda);
    }

    virtual double get_objective() const override {
        return cluster_data::obj / static_cast<double>(g_deg);
    }

    virtual void print(std::ostream& os) const override {
        os << "Normalized LambdaCC: " << get_objective();
    }

    virtual ~normalized_lcc(){}
};