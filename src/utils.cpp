#include "utils.h"

std::pair<Eigen::MatrixXd, double> resample_time_series(const Eigen::MatrixXd& data) {
    int n = data.rows();

    if (n < 2) {
        throw std::runtime_error("Data must have at least 2 points");
    }

    // Извлечение столбцов данных
    Eigen::VectorXd t_old = data.col(0);
    Eigen::VectorXd x_old = data.col(1);
    Eigen::VectorXd y_old = data.col(2);
    Eigen::VectorXd z_old = data.col(3);

    // Вычисление общего времени и шага дискретизации
    double t_start = t_old(0);
    double t_end = t_old(n - 1);
    double T_total = t_end - t_start;
    double dt = T_total / (n - 1);

    // Создание новых равномерных временных меток
    Eigen::VectorXd t_new = Eigen::VectorXd::LinSpaced(n, t_start, t_end);

    // Линейная интерполяция данных с использованием векторизованных операций
    Eigen::VectorXd x_new(n);
    Eigen::VectorXd y_new(n);
    Eigen::VectorXd z_new(n);

    // Находим индексы для интерполяции
    Eigen::VectorXi indices = Eigen::VectorXi::Zero(n);

    // Векторизованное вычисление индексов для интерполяции
    for (int i = 1; i < n; ++i) {
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = (t_new.array() >= t_old(i-1)) && (t_new.array() <= t_old(i));
        indices = (mask).select(i, indices);
    }

    // Векторизованная интерполяция
    for (int i = 0; i < n; ++i) {
        int idx = indices(i);
        if (idx == 0) {
            x_new(i) = x_old(0);
            y_new(i) = y_old(0);
            z_new(i) = z_old(0);
        } else {
            double t0 = t_old(idx-1);
            double t1 = t_old(idx);
            double ratio = (t_new(i) - t0) / (t1 - t0);

            x_new(i) = x_old(idx-1) + ratio * (x_old(idx) - x_old(idx-1));
            y_new(i) = y_old(idx-1) + ratio * (y_old(idx) - y_old(idx-1));
            z_new(i) = z_old(idx-1) + ratio * (z_old(idx) - z_old(idx-1));
        }
    }

    // Формирование результирующего массива
    Eigen::MatrixXd new_data(n, 4);
    new_data.col(0) = t_new;
    new_data.col(1) = x_new;
    new_data.col(2) = y_new;
    new_data.col(3) = z_new;

    return {new_data, dt};
}
