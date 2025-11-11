#pragma once

#include <eigen3/Eigen/Dense>
#include "utils.h"

namespace Measurement {
    /// Partition function
    inline double partition_function(const Eigen::VectorXd& energy_grid, const Eigen::VectorXd& log_dos, double beta) {
        /// Z = \int dE g(E) exp(-beta E)
        Eigen::VectorXd boltzmann_factors(energy_grid.size());
        for (int i = 0; i < energy_grid.size(); ++i) {
            boltzmann_factors[i] = std::exp(-energy_grid[i] * beta + log_dos[i]);
        }
        return utils::trapz(energy_grid, boltzmann_factors);
    }

    /// Internal energy
    inline double internal_energy(const Eigen::VectorXd& energy_grid, const Eigen::VectorXd& log_dos, double beta) {
        /// <E> = (1/Z) \int dE E g(E) exp(-beta E)
        Eigen::VectorXd boltzmann_factors(energy_grid.size());
        for (int i = 0; i < energy_grid.size(); ++i) {
            boltzmann_factors[i] = energy_grid[i] * std::exp(-energy_grid[i] * beta + log_dos[i]);
        }
        double Z = partition_function(energy_grid, log_dos, beta);
        return utils::trapz(energy_grid, boltzmann_factors) / Z;
    }

    /// Specific heat
    inline double specific_heat(const Eigen::VectorXd& energy_grid, const Eigen::VectorXd& log_dos, double beta) {
        /// C = beta^2 ( <E^2> - <E>^2 ) = (1/Z) \int dE E^2 g(E) exp(-beta E) - <E>^2
        Eigen::VectorXd boltzmann_factors(energy_grid.size());
        for (int i = 0; i < energy_grid.size(); ++i) {
            boltzmann_factors[i] = energy_grid[i] * energy_grid[i] * std::exp(-energy_grid[i] * beta + log_dos[i]);
        }
        double Z = partition_function(energy_grid, log_dos, beta);
        double E2 = utils::trapz(energy_grid, boltzmann_factors) / Z;
        double E = internal_energy(energy_grid, log_dos, beta);
        return beta * beta * (E2 - E * E);
    }

    /// Entropy
    inline double entropy(const Eigen::VectorXd& energy_grid, const Eigen::VectorXd& log_dos, double beta) {
        /// S = beta ( <E> - F ) = beta <E> + ln Z
        double Z = partition_function(energy_grid, log_dos, beta);
        double E = internal_energy(energy_grid, log_dos, beta);
        return beta * E + std::log(Z);
    }

    /// Free energy
    inline double free_energy(const Eigen::VectorXd& energy_grid, const Eigen::VectorXd& log_dos, double beta) {
        /// F = - (1/beta) ln Z
        double Z = partition_function(energy_grid, log_dos, beta);
        return - (1.0 / beta) * std::log(Z);
    }
} // namespace Measurement