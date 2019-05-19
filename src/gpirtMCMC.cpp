#include "gpirt.h"

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::mat& y, const int sample_iterations,
                     const int burn_iterations, const double sf,
                     const double ell) {
    int n = y.n_rows;
    int m = y.n_cols;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta and f
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::vec theta = rmvnorm(1, mean_zeros, arma::eye<arma::mat>(n, n)).t();
    arma::mat S = K(theta, theta, sf, ell);
    S.diag() += 0.001;
    arma::mat f = rmvnorm(m, mean_zeros, S).t();
    // Setup grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    int N = theta_star.n_elem;
    arma::mat f_star(N, m);
    // The prior probabilities for theta_star doesn't change between iterations
    arma::vec theta0_prior(N);
    arma::vec thetai_prior(N);
    for ( int i = 0; i < N; ++i ) {
        theta0_prior[i] = d_truncnorm(theta_star[i], 0, 1, R_NegInf, 0, 1);
        thetai_prior[i] = R::dnorm(theta_star[i], 0, 1, 1);
    }
    // K(theta_star, theta_star) also doesn't change between iterations
    arma::mat S11 = K(theta_star, theta_star, sf, ell);
    // Setup results storage
    arma::mat theta_draws(sample_iterations, n);
    arma::cube f_draws(n, m, sample_iterations);
    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Draw new parameter values
        f = draw_f(f, y, S);
        f_star = draw_fstar(f, theta, theta_star, S, S11, sf, ell);
        theta = draw_theta(n, theta_star, y, theta0_prior, thetai_prior, f_star);
        S = K(theta, theta, sf, ell);
        S.diag() += 0.001;
        // Store draws
        if ( iter >= burn_iterations ) {
            theta_draws.row(iter - burn_iterations) = theta.t();
            f_draws.slice(iter - burn_iterations) = f;
        }
    }
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("f", f_draws));
    return result;
}

