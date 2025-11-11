#pragma once

namespace utils {
    inline double trapz(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        double integral = 0.0;
        for (int i = 1; i < x.size(); ++i) {
            integral += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
        }
        return integral;
    }
}