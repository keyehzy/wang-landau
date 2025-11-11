#pragma once

#include "../model.h"

class IsingModel : public Model {
public:
    enum class Init {
        RANDOM,
        ALL_UP,
        ALL_DOWN
    };

    IsingModel(size_t width, size_t height, double J, Init init = Init::RANDOM) : m_width(width), m_height(height), m_J(J), m_init(init) {}

    /// Constructs a Ising model grid with spin configurations according to the specified initialization method.
    Eigen::MatrixXd generate_configuration(std::mt19937& rng) const override {
        switch(m_init) {
            case Init::RANDOM: {
                std::bernoulli_distribution dist(0.5);
                return Eigen::MatrixXd::NullaryExpr(m_width, m_height, [&]() {
                    return dist(rng) ? 1.0 : -1.0;
                });
            }
            case Init::ALL_UP:
                return Eigen::MatrixXd::Ones(m_width, m_height);
            case Init::ALL_DOWN:
                return -Eigen::MatrixXd::Ones(m_width, m_height);
        }
    }

    /// Measure the initial energy of the Ising model grid.
    double measure_energy(const Eigen::MatrixXd& spins) const override {
        double energy = 0;
        for (size_t i = 0; i < (size_t)spins.rows(); ++i) {
            for (size_t j = 0; j < (size_t)spins.cols(); ++j) {
                int spin = spins(i, j);
                int right_spin = spins(i, (j + 1) % spins.cols());
                int down_spin = spins((i + 1) % spins.rows(), j);
                energy -= m_J * spin * (right_spin + down_spin);
            }
        }
        return energy;
    }

    /// Choose a random spin from the grid. Returns its coordinates and the energy change if flipped.
    std::tuple<size_t, double> choose_random_element(const Eigen::MatrixXd& spins, std::mt19937& rng) const override {
        std::uniform_int_distribution<size_t> dist(0, spins.rows() * spins.cols() - 1);
        size_t linear_index = dist(rng);
        size_t row = linear_index / spins.cols();
        size_t col = linear_index % spins.cols();
        
        int spin = spins(row, col);
        int nn_sum =
            spins(row, (col + 1) % spins.cols()) +                // Right neighbor
            spins(row, (col + spins.cols() - 1) % spins.cols()) + // Left neighbor
            spins((row + 1) % spins.rows(), col) +                // Down neighbor
            spins((row + spins.rows() - 1) % spins.rows(), col);  // Up neighbor

        double delta_energy = 2.0 * m_J * spin * nn_sum;
        size_t final_index = row * spins.cols() + col;
        return {final_index, delta_energy};
    }

    /// Flip the spin at the given coordinates.
    void modify_element(Eigen::MatrixXd& spins, size_t index) const override {
        size_t row = index / spins.cols();
        size_t col = index % spins.cols();
        spins(row, col) *= -1.0;
    }

    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    double J() const { return m_J; }

private:
    size_t m_width;
    size_t m_height;
    double m_J;
    Init m_init;
};