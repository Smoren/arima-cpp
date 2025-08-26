#include "arima.h"

ARIMAModel::ARIMAModel(int p, int d, int q) : p(p), d(d), q(q) {
    std::random_device rd;
    generator.seed(rd());
}

void ARIMAModel::fit(const Eigen::VectorXd& series) {
    original_series = series;
    Eigen::VectorXd stationary_series = difference(series, d);

    const size_t n = stationary_series.size();
    const int max_lag = std::max(p, q);

    // Initialize residuals with random values
    std::normal_distribution<double> distribution(0.0, 1.0);
    residuals.resize(static_cast<long>(n));
    for (int i = 0; i < n; ++i) {
        residuals(i) = distribution(generator);
    }

    // Create design matrix
    const size_t rows = n - max_lag;
    Eigen::MatrixXd X(rows, 1 + p + q);
    X.col(0).setOnes();  // Constant term

    // AR components
    for (int i = 0; i < p; ++i) {
        X.col(1 + i) = stationary_series.segment(max_lag - i - 1, rows);
    }

    // MA components
    for (int i = 0; i < q; ++i) {
        X.col(1 + p + i) = residuals.segment(max_lag - i - 1, rows);
    }

    Eigen::VectorXd y = stationary_series.tail(rows);

    // Solve using QR decomposition
    Eigen::VectorXd params = X.colPivHouseholderQr().solve(y);

    ar_params = params.segment(1, p);
    ma_params = params.segment(1 + p, q);

    last_observations = stationary_series.tail(p);
    last_residuals = residuals.tail(q);
}

Eigen::VectorXd ARIMAModel::predict(int steps) {
    if (ar_params.size() == 0 || ma_params.size() == 0) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    Eigen::VectorXd forecasts(steps);
    Eigen::VectorXd current_obs = last_observations;
    Eigen::VectorXd current_resids = last_residuals;

    for (int i = 0; i < steps; ++i) {
        const double ar_value = ar_params.reverse().dot(current_obs);
        const double ma_value = ma_params.reverse().dot(current_resids);
        forecasts(i) = ar_value + ma_value;

        // Update observations and residuals
        current_obs = (Eigen::VectorXd(current_obs.size() + 1) << current_obs, forecasts(i)).finished().tail(p);
        current_resids = (Eigen::VectorXd(current_resids.size() + 1) << current_resids, 0).finished().tail(q);
    }

    if (d > 0) {
        Eigen::VectorXd initial = original_series.tail(d);
        forecasts = inverse_difference(forecasts, initial, d);
    }

    return forecasts;
}

Eigen::VectorXd ARIMAModel::difference(const Eigen::VectorXd& series, int n) {
    Eigen::VectorXd result = series;
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd diff(result.size() - 1);
        for (int j = 0; j < result.size() - 1; ++j) {
            diff(j) = result(j + 1) - result(j);
        }
        result = diff;
    }
    return result;
}

Eigen::VectorXd ARIMAModel::inverse_difference(const Eigen::VectorXd& series, const Eigen::VectorXd& initial, int n) {
    Eigen::VectorXd result = series;
    Eigen::VectorXd init = initial;
    for (int i = 0; i < n; ++i) {
        size_t length = result.size();
        Eigen::VectorXd cumsum(length + init.size());
        cumsum.segment(0, init.size()) = init;

        for (int j = 0; j < length; ++j) {
            cumsum(init.size() + j) = cumsum(init.size() + j - 1) + result(j);
        }
        result = cumsum.tail(length + init.size() - n + i);
        init = init.head(init.size() - 1);
    }
    return result;
}