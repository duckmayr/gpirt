#include "gpirt.h"

Rcpp::List gpirtMCMC(const arma::mat& y, const int sample_iterations,
                     const int burn_iterations) {
    int n y.n_rows;
    int m y.n_cols;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta and f
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::vec theta = rmvnorm(1, mean_zeros, arma::eye<arma::mat>(n, n)).t();
    arma::mat S = K(theta, theta, 1.0, 1.0);
    arma::mat f = rmvnorm(m, mean_zeros, S).t();
    // Setup grids
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::mat f_star(theta_star.n_elem, m);
    // Setup results storage
    arma::mat theta_draws(sample_iterations, n);
    arma::cube f_draws(n, m, sample_iterations);
    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Draw f and fstar
        for ( int j = 0; j < m; ++j ) {
            f.col(j) = draw_f(f.col(j), y.col(j), S);
            f_star.col(j) = draw_fstar(f.col(j), theta, theta_star);
        }
        // Draw theta
        theta = draw_theta(theta_star, f_star);
        // Store draws
        if ( iter >= burn_iterations ) {
            theta_draws.row(iter - burn_iterations) = theta.t();
            f_draws.slice(iter - burn_iterations) = f;
        }
    }
    Rcpp::List result = Rcpp::List::create(Rcpp::named("theta", theta_draws),
                                           Rcpp::named("f", f_draws));
    return result;
}

