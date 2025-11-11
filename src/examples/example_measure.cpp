#include "../wang_landau.h"
#include "ising_model.h
#include "measure.h"

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

    auto beta_grid = Eigen::VectorXd::LinSpaced(1000, 0.1, 10.0);
    for (double beta : beta_grid) {
        double I = Measurement::internal_energy(energy_grid, stats.g, beta);
        double C = Measurement::specific_heat(energy_grid, stats.g, beta);
        double S = Measurement::entropy(energy_grid, stats.g, beta);
        double F = Measurement::free_energy(energy_grid, stats.g, beta);
        std::cout << beta << " " << I << " " << C << " " << S << " " << F << "\n";;
    }
    return 0;
}