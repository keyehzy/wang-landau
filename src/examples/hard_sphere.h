#pragma once

#include <eigen3/Eigen/Dense>
#include <tuple>
#include <random>
#include <cmath>
#include <stdexcept>
#include <numbers>

#include "../model.h"

class HardSphereModel : public Model {
public:
    enum class Init {
        RANDOM,
        NON_OVERLAPPING,
        SQUARE_LATTICE,
        TRIANGULAR_LATTICE,
    };

    HardSphereModel(size_t n_particles, double box_length, double radius, double max_displacement, Init init = Init::RANDOM)
        : n_particles(n_particles),
          box_length(box_length),
          radius(radius),
          max_displacement(max_displacement),
          init(init),
          proposed_index_(0),
          proposed_new_pos_(Eigen::Vector2d::Zero()) {}

    /// Generate a configuration
    Eigen::MatrixXd generate_configuration(RNGType& rng) const override {
        Eigen::MatrixXd config(n_particles, 2);
        switch (init) {
            case Init::RANDOM: {
                std::uniform_real_distribution<double> dist(0.0, box_length);
                return Eigen::MatrixXd::NullaryExpr(n_particles, 2, [&]() {
                    return dist(rng);
                });
            }

            case Init::NON_OVERLAPPING: {
                std::uniform_real_distribution<double> dist(0.0, n_particles);
                const size_t max_trials = 1e6;

                for (size_t i = 0; i < n_particles; ++i) {
                    bool placed = false;
                    for (size_t trial = 0; trial < max_trials; ++trial) {
                        Eigen::Vector2d pos;
                        pos[0] = dist(rng);
                        pos[1] = dist(rng);

                        if (!has_overlap_with_previous(config, i, pos)) {
                            config(i, 0) = pos[0];
                            config(i, 1) = pos[1];
                            placed = true;
                            break;
                        }
                    }
                    if (!placed) {
                        throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles without overlap.");
                    }
                }

                return config;
            }

            case Init::SQUARE_LATTICE: {
                const double d = std::max(0.0, 2.0 * radius - eps);
                const double a   = d;
                const double pad = radius;
                
                if (a <= 0.0 || box_length <= 2.0*pad) {
                        throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a square lattice");
                }

                const int nx = static_cast<int>(std::floor((box_length - 2.0*pad) / a)) + 1;
                const int ny = static_cast<int>(std::floor((box_length - 2.0*pad) / a)) + 1;

                if (nx <= 0 || ny <= 0) {
                    throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a square lattice");
                }

                const size_t cap = static_cast<size_t>(nx) * static_cast<size_t>(ny);
                
                if (cap < n_particles) {
                    throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a square lattice");
                }

                size_t k = 0;
                for (int j = 0; j < ny && k < n_particles; ++j) {
                    const double y = pad + j * a;
                    for (int i = 0; i < nx && k < n_particles; ++i) {
                        const double x = pad + i * a;
                        config(k, 0) = x;
                        config(k, 1) = y;
                        ++k;
                    }
                }
                return config;
            }

            case Init::TRIANGULAR_LATTICE: {
                const double d = std::max(0.0, 2.0 * radius - eps);
                const double a   = d;
                const double b   = 0.5 * std::sqrt(3.0) * d;
                const double pad = radius;
                
                if (a <= 0.0 || b <= 0.0 || box_length <= 2.0*pad) {
                    throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a triangular lattice");
                }

                const int nx_even = static_cast<int>(std::floor((box_length - 2.0*pad) / a)) + 1;
                const int nx_odd  = static_cast<int>(std::floor((box_length - 2.0*pad - 0.5*a) / a)) + 1;
                const int ny      = static_cast<int>(std::floor((box_length - 2.0*pad) / b)) + 1;

                if (ny <= 0 || nx_even <= 0) {
                    throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a triangular lattice");
                }

                // total capacity with staggering
                size_t cap = 0;
                for (int j = 0; j < ny; ++j) {
                    const bool odd = (j % 2) == 1;
                    const int nxj = odd ? std::max(0, nx_odd) : nx_even;
                    cap += static_cast<size_t>(nxj);
                }
                
                if (cap < n_particles) {
                    throw std::runtime_error("HardSphereModel::generate_configuration: failed to place all particles in a triangular lattice");
                }

                size_t k = 0;
                for (int j = 0; j < ny && k < n_particles; ++j) {
                    const bool odd = (j % 2) == 1;
                    const int nxj  = odd ? std::max(0, nx_odd) : nx_even;
                    const double y = pad + j * b;
                    for (int i = 0; i < nxj && k < n_particles; ++i) {
                        const double x = pad + i * a + (odd ? 0.5 * a : 0.0);
                        config(k, 0) = x;
                        config(k, 1) = y;
                        ++k;
                    }
                }
                return config;
            }
        }
    }

    /// Energy = number of overlapping pairs (i < j).
    double measure_energy(const Eigen::MatrixXd& configuration) const override {
        double energy = 0.0;

        for (size_t i = 0; i < n_particles; ++i) {
            Eigen::Vector2d ri = configuration.row(i);
            for (size_t j = i + 1; j < n_particles; ++j) {
                Eigen::Vector2d rj = configuration.row(j);
                if (overlap(ri, rj)) {
                    energy += 1.0;
                }
            }
        }

        return energy;
    }

    /// Propose moving a random particle by a small displacement.
    std::tuple<size_t, double> choose_random_element(const Eigen::MatrixXd& configuration, RNGType& rng) const override {
        // Choose random particle
        std::uniform_int_distribution<size_t> pick(0, n_particles - 1);
        size_t i = pick(rng);

        // Propose random displacement in [-max_displacement, max_displacement]
        std::uniform_real_distribution<double> disp(-max_displacement, max_displacement);
        Eigen::Vector2d old_pos = configuration.row(i);
        Eigen::Vector2d new_pos = old_pos + Eigen::Vector2d(disp(rng), disp(rng));

        // Apply periodic boundary conditions
        new_pos = apply_pbc(new_pos);

        double count_old = 0;
        double count_new = 0;

        for (size_t j = 0; j < n_particles; ++j) {
            if (j == i) continue;

            Eigen::Vector2d rj = configuration.row(j);

            bool old_overlap = overlap(old_pos, rj);
            bool new_overlap = overlap(new_pos, rj);

            if (old_overlap) {
                count_old += 1.0;
            }

            if (new_overlap) {
                count_new += 1.0;
            }
        }

        // Compute local ΔE from pairs involving particle i only.
        double delta_energy = count_new - count_old;

        // Store proposal for possible later commit
        proposed_index_   = i;
        proposed_new_pos_ = new_pos;

        return {i, delta_energy};
    }

    /// Commit the last proposed move for the given index.
    void modify_element(Eigen::MatrixXd& configuration, size_t) const override {
        configuration(proposed_index_, 0) = proposed_new_pos_[0];
        configuration(proposed_index_, 1) = proposed_new_pos_[1];
    }

    double max_energy() const {
        return 0.5 * n_particles * (n_particles - 1);
    }

    double min_energy() {
        return 0.0;
    }

    // Render the configuration as fixed-size ASCII art to an output stream.
    // The box is drawn with a border. Coordinates are wrapped with PBC and
    // the y-axis is shown upward (origin at bottom-left).
void print_ascii(const Eigen::MatrixXd& configuration, std::ostream& os = std::cout) const {
    size_t width = 60;
    size_t height = 30;

    // Occupancy grid (counts per cell)
    std::vector<std::vector<unsigned int>> grid(height, std::vector<unsigned int>(width, 0));

    // Bin particles into cells
    for (size_t i = 0; i < n_particles; ++i) {
        // Wrap to [0, L)
        Eigen::Vector2d p(configuration(i, 0), configuration(i, 1));
        p = apply_pbc(p);

        // Map to [0, width/height-1]
        // Use min(...) guard to avoid rounding up to width/height
        size_t ix = std::min<size_t>(width  - 1, static_cast<size_t>(std::floor(p[0] / box_length * width)));
        size_t iy = std::min<size_t>(height - 1, static_cast<size_t>(std::floor(p[1] / box_length * height)));

        // Invert y so row 0 prints at the top but corresponds to highest y
        // (i.e., origin at bottom-left visually)
        size_t row = (height - 1) - iy;
        size_t col = ix;

        grid[row][col] += 1u;
    }

    // Mapping from occupancy count to a character
    auto glyph = [](unsigned int c) -> char {
        if (c == 0) return ' ';
        if (c == 1) return '.';
        if (c <= 3) return 'o';
        if (c <= 6) return 'O';
        if (c <= 9) return '0';
        return '@'; // 10+ in the same cell
    };

    // Top border
    os << '+' << std::string(width, '-') << "+\n";

    // Rows
    for (size_t r = 0; r < height; ++r) {
        os << '|';
        for (size_t c = 0; c < width; ++c) {
            os << glyph(grid[r][c]);
        }
        os << "|\n";
    }

    // Bottom border
    os << '+' << std::string(width, '-') << "+\n";

    // Optional legend
    os << "N=" << n_particles
       << "  L=" << box_length
       << "  r=" << radius
       << "  packing=" << (double)(n_particles * std::numbers::pi_v<double> * radius * radius) / (double)(box_length * box_length)
       << "  symbols: '.'=1, 'o'=2-3, 'O'=4-6, '0'=7-9, '@'=10+\n";
}


private:
    size_t n_particles;
    double box_length;
    double radius;
    double max_displacement;
    Init init;

    const double eps = 1e-6;

    // Mutable to support proposal storage in const choose_random_element.
    mutable size_t    proposed_index_;
    mutable Eigen::Vector2d proposed_new_pos_;

    // Minimum-image distance squared under PBC.
    double distance2_pbc(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];

        dx -= std::round(dx / box_length) * box_length;
        dy -= std::round(dy / box_length) * box_length;

        return dx * dx + dy * dy;
    }

    bool overlap(const Eigen::Vector2d& a, const Eigen::Vector2d& b) const {
        double contact2 = (2.0 * radius - eps) * (2.0 * radius - eps);
        return distance2_pbc(a, b) < contact2 - eps;
    }

    bool has_overlap_with_previous(const Eigen::MatrixXd& config, size_t count, const Eigen::Vector2d& pos) const {
        for (size_t j = 0; j < count; ++j) {
            Eigen::Vector2d rj = config.row(j);
            if (overlap(pos, rj)) {
                return true;
            }
        }
        return false;
    }

    // Apply periodic boundary conditions to keep coordinates in [0, L).
    Eigen::Vector2d apply_pbc(const Eigen::Vector2d& pos) const {
        Eigen::Vector2d wrapped = pos;
        // Use fmod-style wrap; ensure result in [0, L).
        for (int d = 0; d < 2; ++d) {
            double x = wrapped[d];
            x = std::fmod(x, box_length);
            if (x < 0.0) x += box_length;
            wrapped[d] = x;
        }
        return wrapped;
    }
};