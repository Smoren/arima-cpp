#pragma once

#include <Eigen/Dense>

std::pair<Eigen::MatrixXd, double> resample_time_series(const Eigen::MatrixXd& data);
