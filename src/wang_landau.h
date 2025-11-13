#pragma once

#include <eigen3/Eigen/Dense>
#include <random>
#include <tuple>
#include <fstream>
#include <cassert>
#include <limits>

/// Monte Carlo statistics. In the Wang-Landau algorithm, we track the histogram and the density of states.
struct MCStats {
    MCStats(size_t num_bins) : histogram(Eigen::VectorXd::Zero(num_bins)), g(Eigen::VectorXd::Zero(num_bins)) {}

    double histogram_flatness() const {
        double mean = histogram.mean();
        if (mean == 0.0) return 0.0;
        double hmin = histogram.minCoeff();
        return hmin / mean;
    }

    /// Normalize the density of states by deleting by the lowest value.
    void normalize() {
        double min_g = g.minCoeff();
        g = g.array() - min_g;
    }

    void save() const {
        std::ofstream hist_file("histogram.dat");
        hist_file << histogram << std::endl;
        hist_file.close();
        std::ofstream g_file("g.dat");
        g_file << g << std::endl;
        g_file.close();
    }

    Eigen::VectorXd histogram;
    Eigen::VectorXd g;
};

template <typename Configuration, typename Model>
MCStats monte_carlo(Configuration conf , const Eigen::VectorXd& energy_grid, const Model& model, std::mt19937& rng) {
    double max_energy = energy_grid.maxCoeff();
    double min_energy = energy_grid.minCoeff();
    size_t num_bins   = energy_grid.size();
    double bin_width  = (max_energy - min_energy) / (num_bins - 1);
    MCStats stats(num_bins);
    
    double current_energy = model.measure_energy(conf);
    std::uniform_real_distribution<> dist(0.0, 1.0);
    double factor = 1.0;

    for (size_t step = 0; /*loop*/ ; ++step) {
        auto [index, delta_energy] = model.choose_random_element(conf, rng);
        double new_energy = current_energy + delta_energy;

        assert(current_energy >= min_energy && current_energy <= max_energy);
        assert(new_energy >= min_energy && new_energy <= max_energy);

        size_t current_bin = static_cast<size_t>((current_energy - min_energy) / bin_width);
        size_t new_bin = static_cast<size_t>((new_energy - min_energy) / bin_width);

        assert(current_bin < (size_t)stats.g.size());
        assert(new_bin < (size_t)stats.g.size());

        // Metropolis criterion
        if (stats.g[new_bin] < stats.g[current_bin] || std::exp(stats.g[current_bin] - stats.g[new_bin]) > dist(rng)) {
            model.modify_element(conf, index);
            current_bin = new_bin;
            current_energy = new_energy;
        }

        /// log_g(E) <-- log_g(E) + f
        /// H(E) <-- H(E) + 1
        stats.histogram[current_bin] += 1;
        stats.g[current_bin] += factor;

        if (step % 10000 == 0) {
            if (stats.histogram_flatness() >= 0.8) {
                stats.histogram.setZero();
                factor *= 0.5;
                if (factor < 1e-6) {
                    break;
                }
            }
        }
    }

    stats.normalize();
    return stats;
}