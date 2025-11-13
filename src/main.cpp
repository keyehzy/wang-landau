#include <iostream>

#include "wang_landau.h"
#include "examples/ising_model.h"
#include "examples/hard_sphere.h"
#include "examples/measure.h"

// TODO:
// Parallelize energy grid with overlaping regions
// Compute dyanmical correlation functions

int main() {
    double radius = 1.0;
    auto model = HardSphereModel(25, 10.0, radius, radius / 2.0, HardSphereModel::Init::RANDOM);
    std::mt19937 rng(std::random_device{}());
    auto conf = model.generate_configuration(rng);
    model.print_ascii(conf);
    
    double max_energy = model.max_energy();
    double min_energy = std::max(model.min_energy(), 5.0);
    double delta_energy = 1.0;
    size_t num_bins = (size_t)((max_energy - min_energy) / delta_energy) + 1;
    auto energy_grid = Eigen::VectorXd::LinSpaced(num_bins, min_energy, max_energy);
    auto stats = monte_carlo(conf, energy_grid, model, rng);
    
    // Eigen::MatrixXd output(energy_grid.size(), 2);
    // output << energy_grid, stats.g;
    // std::cout << output << std::endl;

    auto beta_grid = Eigen::VectorXd::LinSpaced(1000, -3.0, 3.0);
    auto log_beta_grid = beta_grid.array().exp();
    for (double beta : log_beta_grid) {
        double I = Measurement::internal_energy(energy_grid, stats.g, beta);
        double C = Measurement::specific_heat(energy_grid, stats.g, beta);
        double S = Measurement::entropy(energy_grid, stats.g, beta);
        double F = Measurement::free_energy(energy_grid, stats.g, beta);
        std::cout << beta << " " << I << " " << C << " " << S << " " << F << "\n";;
    }
    return 0;
}