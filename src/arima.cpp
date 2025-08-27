#include "arima.h"

#include <iostream>

#include <chrono>

ARIMAModel::ARIMAModel(int p, int d, int q) : p(p), d(d), q(q) {}

void ARIMAModel::fit(const Eigen::VectorXd& series) {
    // Сохраняем исходные данные
    original_series = series;

    // Выполняем дифференцирование
    Eigen::VectorXd stationary_series = difference(series, d); // TODO первые члены неадекватные

    // Создаем матрицы признаков для AR и MA компонент
    const Eigen::Index n = stationary_series.size();
    const Eigen::Index max_lag = std::max(p, q);

    if (n <= max_lag) {
        throw std::invalid_argument("Series too short for given ARIMA parameters");
    }

    // Создаем матрицу признаков для AR части
    Eigen::MatrixXd X_ar(n - max_lag, p);
    for (int i = 0; i < p; ++i) {
        X_ar.col(i) = stationary_series.segment(max_lag - i - 1, n - max_lag);
    }

    // Создаем матрицу признаков для MA части
    Eigen::MatrixXd X_ma(n - max_lag, q);
    residuals = Eigen::VectorXd::Random(n); // Инициализируем случайными остатками
    for (int i = 0; i < q; ++i) {
        X_ma.col(i) = residuals.segment(max_lag - i - 1, n - max_lag);
    }

    // Комбинируем признаки
    Eigen::MatrixXd X(n - max_lag, 1 + p + q);
    X.leftCols(1).fill(1.0);  // Константа
    X.block(0, 1, n - max_lag, p) = X_ar;
    X.block(0, 1 + p, n - max_lag, q) = X_ma;

    Eigen::VectorXd y = stationary_series.segment(max_lag, n - max_lag);

    // Решаем систему линейных уравнений методом наименьших квадратов
    // Eigen::VectorXd params = X.colPivHouseholderQr().solve(y);

    constexpr double lambda = 1e-8;  // малое значение для регуляризации
    Eigen::VectorXd params = (X.transpose() * X + lambda * Eigen::MatrixXd::Identity(X.cols(), X.cols())).ldlt().solve(X.transpose() * y);

    // Разделяем параметры на AR и MA компоненты
    ar_params = params.segment(1, p);
    ma_params = params.segment(1 + p, q);

    // Сохраняем последние наблюдения и остатки для прогнозирования
    last_observations = stationary_series.tail(p);
    last_residuals = residuals.tail(q);
}

Eigen::VectorXd ARIMAModel::predict(int steps) {
    if (ar_params.size() == 0 || ma_params.size() == 0) {
        throw std::runtime_error("Model must be fitted before prediction");
    }

    Eigen::VectorXd forecasts = Eigen::VectorXd::Zero(steps);
    Eigen::VectorXd current_obs = last_observations;
    Eigen::VectorXd current_resids = last_residuals;

    for (int i = 0; i < steps; ++i) {
        // AR компонента (векторизованная)
        const double ar_value = ar_params.dot(current_obs.reverse());

        // MA компонента (векторизованная)
        const double ma_value = ma_params.dot(current_resids.reverse());

        // Константа и сумма компонент
        const double forecast_val = ar_value + ma_value;
        forecasts(i) = forecast_val;

        // Обновляем наблюдения и остатки
        current_obs = current_obs.tail(p - 1);
        current_obs.conservativeResize(p);
        current_obs(p - 1) = forecast_val;

        current_resids = current_resids.tail(q - 1);
        current_resids.conservativeResize(q);
        current_resids(q - 1) = 0.0;  // Предполагаем нулевые будущие остатки
    }

    // Преобразуем обратно к исходному масштабу
    if (d > 0) {
        // Для обратного дифференцирования используем последние значения исходного ряда
        Eigen::VectorXd initial_values = original_series.tail(d);
        forecasts = inverse_difference(forecasts, initial_values, d);
    }

    return forecasts.head(steps);
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
    Eigen::VectorXd current_initial = initial;

    for (int i = 0; i < n; ++i) {
        // Векторизованное обратное преобразование
        Eigen::VectorXd cumsum = Eigen::VectorXd::Zero(result.size() + current_initial.size());
        cumsum.head(current_initial.size()) = current_initial;
        cumsum.tail(result.size()) = result;

        // Суммируем элементы по строкам
        for (int j = 1; j < cumsum.size(); ++j) {
            cumsum(j) += cumsum(j-1);
        }

        result = cumsum.tail(result.size());
        current_initial = current_initial.head(current_initial.size() - 1);
    }
    return result;
}
