#include <iostream>
#include <random>
#include <cassert>
#include <fstream>
#include <eigen3/Eigen/Dense>

#include "examples/ising_model.h"
#include "examples/measure.h"

/// Monte Carlo statistics. In the Wang-Landau algorithm, we track the histogram and the density of states.
struct MCStats {
    MCStats(size_t num_bins) : histogram(Eigen::VectorXd::Zero(num_bins)), g(Eigen::VectorXd::Zero(num_bins)) {}

    /// Compute the flatness of the histogram as the maximum relative deviation from the mean.
    double histogram_flatness() const {
        double mean = histogram.mean();

        if (mean == 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        
        double max_rel_dev = 0.0;

        for (int i = 0; i < histogram.size(); ++i) {
            if (histogram[i] > 0) { 
                double rel_dev = std::abs(histogram[i] - mean) / mean;
                if (rel_dev > max_rel_dev) {
                    max_rel_dev = rel_dev;
                }
            }
        }

        return max_rel_dev;
    }

    /// Try to flatten the histogram. If flat enough, reset histogram and reduce factor.
    /// Return wheter the histogram was flattened.
    bool try_flatten_histogram() {
        if (histogram_flatness() < 0.2) {
            histogram.setZero();
            return true;
        }
        return false;
    }

    /// Fill in missing values in the density of states using linear interpolation.
    void interpolate() {
        for (int i = 0; i < g.size(); ++i) {
            if (g[i] == 0.0) {
                int left = i - 1;
                while (left >= 0 && g[left] == 0.0) {
                    --left;
                }
                int right = i + 1;
                while (right < g.size() && g[right] == 0.0) {
                    ++right;
                }

                if (left >= 0 && right < g.size()) {
                    g[i] = g[left] + (g[right] - g[left]) * (i - left) / (right - left);
                } else if (left >= 0) {
                    g[i] = g[left];
                } else if (right < g.size()) {
                    g[i] = g[right];
                }
            }
        }
    }

    /// Normalize the density of states by deleting by the lowest value.
    void normalize() {
        double min_g = g.minCoeff();
        g = g.array() - min_g;
    }

    /// Finalize
    void finilize() {
        interpolate();
        normalize();
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

MCStats monte_carlo(Eigen::MatrixXd spins, const Eigen::VectorXd& energy_grid, const IsingModel& model) {
    double max_energy = energy_grid.maxCoeff();
    double min_energy = energy_grid.minCoeff();
    size_t num_bins   = energy_grid.size();
    double bin_width  = (max_energy - min_energy) / (num_bins - 1);
    assert(std::abs(bin_width - 4.0 * model.J()) < 1e-6);
    MCStats stats(num_bins);
    
    double current_energy = model.measure_energy(spins);
    double factor = 1.0;

    for (size_t step = 0; /*loop*/ ; ++step) {
        auto [row, col, delta_energy] = model.choose_random_spin(spins);
        double new_energy = current_energy + delta_energy;

        assert(current_energy >= min_energy && current_energy <= max_energy);
        assert(new_energy >= min_energy && new_energy <= max_energy);

        size_t current_bin = static_cast<size_t>((current_energy - min_energy) / bin_width);
        size_t new_bin = static_cast<size_t>((new_energy - min_energy) / bin_width);

        assert(current_bin < (size_t)stats.g.size());
        assert(new_bin < (size_t)stats.g.size());

        // Metropolis criterion
        if (stats.g[new_bin] < stats.g[current_bin] ||
            std::exp(stats.g[current_bin] - stats.g[new_bin]) > std::uniform_real_distribution<>(0.0, 1.0)(rng)) {
            model.flip_spin(spins, row, col);
            current_bin = new_bin;
            current_energy = new_energy;
        }

        /// log_g(E) <-- log_g(E) + f
        /// H(E)    <-- H(E) + 1
        stats.histogram[current_bin] += 1;
        stats.g[current_bin] += factor;

        if (step % 10000 == 0) {
            if (stats.try_flatten_histogram()) {
                factor *= 0.5;
            }
            if (factor < 1e-6) {
                break;
            }
        }
    }

    return stats;
}

int main() {
    auto model = IsingModel(16, 16, 1.0, IsingModel::Init::ALL_UP);
    auto spins = model.generate_spin_configuration();
    
    double max_energy = 2.0 * model.J() * spins.size();
    double min_energy = -max_energy;
    double dE = 4.0 * model.J();
    size_t num_bins = static_cast<size_t>((max_energy - min_energy) / dE) + 1;

    auto energy_grid = Eigen::VectorXd::LinSpaced(num_bins, min_energy, max_energy);

    auto stats = monte_carlo(spins, energy_grid, model);
    stats.finilize();
    stats.save();

    auto beta_grid = Eigen::VectorXd::LinSpaced(1000, 0.1, 10.0);
    // auto logspace_beta_grid = beta_grid.array().exp();
    for (double beta : beta_grid) {
        double I = Measurement::internal_energy(energy_grid, stats.g, beta);
        double C = Measurement::specific_heat(energy_grid, stats.g, beta);
        double S = Measurement::entropy(energy_grid, stats.g, beta);
        double F = Measurement::free_energy(energy_grid, stats.g, beta);
        std::cout << beta << " " << I << " " << C << " " << S << " " << F << "\n";;
    }
    return 0;
}