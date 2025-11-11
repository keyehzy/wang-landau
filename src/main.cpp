#include <iostream>

#include "wang_landau.h"
#include "examples/ising_model.h"

int main() {
    auto model = IsingModel(16, 16, 1.0, IsingModel::Init::ALL_UP);
    std::mt19937 rng(42);
    auto spins = model.generate_configuration(rng);
    
    double max_energy = 2.0 * model.J() * spins.size();
    double min_energy = -max_energy;
    double dE = 4.0 * model.J();
    size_t num_bins = static_cast<size_t>((max_energy - min_energy) / dE) + 1;
    auto energy_grid = Eigen::VectorXd::LinSpaced(num_bins, min_energy, max_energy);

    auto stats = monte_carlo(spins, energy_grid, model, rng);
    stats.finalize();
    
    Eigen::MatrixXd output(energy_grid.size(), 2);
    output << energy_grid, stats.g;
    std::cout << output << std::endl;
    return 0;
}