#pragma once

#include <Eigen/Dense>
#include <random>

class ARIMAModel {
public:
    explicit ARIMAModel(int p, int d, int q);

    void fit(const Eigen::VectorXd& series);

    Eigen::VectorXd predict(int steps);

private:
    int p, d, q;
    Eigen::VectorXd ar_params;
    Eigen::VectorXd ma_params;
    Eigen::VectorXd residuals;
    Eigen::VectorXd last_observations;
    Eigen::VectorXd last_residuals;
    Eigen::VectorXd original_series;

    static Eigen::VectorXd difference(const Eigen::VectorXd& series, int n);

    static Eigen::VectorXd inverse_difference(const Eigen::VectorXd& series, const Eigen::VectorXd& initial, int n);
};
