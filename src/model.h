#pragma once

#include <eigen3/Eigen/Dense>
#include <tuple>
#include <random>

/// Interface for models used in the Wang-Landau algorithm.
class Model {
public:
    virtual ~Model() = default;

    /// Type of the random number generator used in the model.
    using RNGType = std::mt19937;

    /// Generate a configuration for the model.
    virtual Eigen::MatrixXd generate_configuration(RNGType& rng) const = 0;
    
    /// Measure the energy of a given configuration.
    virtual double measure_energy(const Eigen::MatrixXd& configuration) const = 0;

    /// Choose a random element from the configuration. Returns its coordinates and the energy change if modified.
    /// The coordinates should be return in linear form (i.e., single index) and the user should be able to map it 
    // back to 2D if needed.
    virtual std::tuple<size_t, double> choose_random_element(const Eigen::MatrixXd& configuration, RNGType& rng) const = 0;

    /// Modify the element at the given coordinates.
    virtual void modify_element(Eigen::MatrixXd& configuration, size_t index) const = 0;
};

    