#include "gpirt.h"

// [[Rcpp::export(.gpirtMCMC0)]]
Rcpp::List gpirtMCMC0(const arma::mat& y, arma::vec theta,
                      const int sample_iterations, const int burn_iterations,
                      const arma::vec& means, const arma::uvec& groups,
                      const double sf, const double ell,
                      const arma::mat& beta_prior_means,
                      const arma::mat& beta_prior_sds,
                      const arma::mat& beta_step_sizes) {
    int n = y.n_rows;
    int m = y.n_cols;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta, f, and beta
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::mat S = K(theta, theta, sf, ell);
    S.diag() += 0.001;
    arma::mat f = rmvnorm(m, mean_zeros, S).t();
    arma::mat beta(2, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword p = 0; p < 2; ++p ) {
            beta(p, j) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
        }
    }
    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean
    arma::mat X(n, 2);
    X.col(0) = arma::ones<arma::vec>(n);
    X.col(1) = theta;
    arma::mat mu = X * beta;
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    int N = theta_star.n_elem;
    arma::mat f_star(N, m);
    arma::mat Xstar(N, 2);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    arma::mat mu_star = Xstar * beta;
    // The prior probabilities for theta_star doesn't change between iterations
    int num_groups = groups.n_elem;
    arma::mat theta_prior(N, num_groups);
    for ( arma::uword g = 0; g < num_groups; ++g ) {
        for ( arma::uword k = 0; k < N; ++k ) {
            theta_prior(k, g) = R::dnorm(theta_star[k], means[g], 1.0, 1);
        }
    }
    // Setup results storage
    arma::mat theta_draws(sample_iterations + 1, n);
    arma::cube beta_draws(2, m, sample_iterations + 1);
    arma::cube f_draws(n, m, sample_iterations + 1);
    arma::cube fstar_draws(N, m, sample_iterations + 1);
    // Store initial values
    theta_draws.row(0)   = theta.t();
    beta_draws.slice(0)  = beta;
    f_draws.slice(0)     = f;
    fstar_draws.slice(0) = f_star;
    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;
    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Update progress and check for user interrupt (normally you'd do this
        // and the interrupt check less often, but -- at least for now -- each
        // iteration takes long enough to warrant doing it each time)
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
        // Draw new parameter values
        f = draw_f(f, y, S, mu);
        f_star = draw_fstar(f, theta, theta_star, S, sf, ell, mu, mu_star);
        theta = draw_theta(n, theta_star, y, theta_prior, groups, f_star,
                           mu_star);
        X.col(1) = theta;
        beta = draw_beta(beta, X, y, f, beta_prior_means, beta_prior_sds,
                         beta_step_sizes);
        mu = X * beta;
        mu_star = Xstar * beta;
        S = K(theta, theta, sf, ell);
        S.diag() += 0.001;
        // Store draws
        if ( iter >= burn_iterations ) {
            theta_draws.row(iter - burn_iterations + 1) = theta.t();
            beta_draws.slice(iter - burn_iterations + 1) = beta;
            f_draws.slice(iter - burn_iterations + 1) = f;
            fstar_draws.slice(iter - burn_iterations + 1) = f_star;
        }
    }
    Rprintf("\r100.000 %% complete\n");
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("beta", beta_draws),
                                           Rcpp::Named("f", f_draws),
                                           Rcpp::Named("fstar", fstar_draws));
    return result;
}

// [[Rcpp::export(.gpirtMCMC1)]]
Rcpp::List gpirtMCMC1(const arma::mat& y, arma::vec theta,
                      const int sample_iterations, const int burn_iterations,
                      const arma::vec& means, const arma::uvec& groups,
                      const double sf, const double ell,
                      const arma::mat& beta_prior_means,
                      const arma::mat& beta_prior_sds,
                      const arma::mat& beta_step_sizes) {
    int n = y.n_rows;
    int m = y.n_cols;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta, f, and beta
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::mat S = K(theta, theta, sf, ell);
    S.diag() += 0.001;
    arma::mat f = rmvnorm(m, mean_zeros, S).t();
    arma::mat beta(2, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword p = 0; p < 2; ++p ) {
            beta(p, j) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
        }
    }
    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean
    arma::mat X(n, 2);
    X.col(0) = arma::ones<arma::vec>(n);
    X.col(1) = theta;
    arma::mat mu = X * beta;
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    int N = theta_star.n_elem;
    arma::mat f_star(N, m);
    arma::mat Xstar(N, 2);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    arma::mat mu_star = Xstar * beta;
    // The prior probabilities for theta_star doesn't change between iterations
    int num_groups = groups.n_elem;
    arma::mat theta_prior(N, num_groups);
    for ( arma::uword g = 0; g < num_groups; ++g ) {
        for ( arma::uword k = 0; k < N; ++k ) {
            theta_prior(k, g) = R::dnorm(theta_star[k], means[g], 1.0, 1);
        }
    }
    // Setup results storage
    arma::mat theta_draws(sample_iterations + 1, n);
    arma::cube beta_draws(2, m, sample_iterations + 1);
    arma::cube f_draws(n, m, sample_iterations + 1);
    // Store initial values
    theta_draws.row(0)   = theta.t();
    beta_draws.slice(0)  = beta;
    f_draws.slice(0)     = f;
    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;
    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Update progress and check for user interrupt (normally you'd do this
        // and the interrupt check less often, but -- at least for now -- each
        // iteration takes long enough to warrant doing it each time)
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
        // Draw new parameter values
        f = draw_f(f, y, S, mu);
        f_star = draw_fstar(f, theta, theta_star, S, sf, ell, mu, mu_star);
        theta = draw_theta(n, theta_star, y, theta_prior, groups, f_star,
                           mu_star);
        X.col(1) = theta;
        beta = draw_beta(beta, X, y, f, beta_prior_means, beta_prior_sds,
                         beta_step_sizes);
        mu = X * beta;
        mu_star = Xstar * beta;
        S = K(theta, theta, sf, ell);
        S.diag() += 0.001;
        // Store draws
        if ( iter >= burn_iterations ) {
            theta_draws.row(iter - burn_iterations + 1) = theta.t();
            beta_draws.slice(iter - burn_iterations + 1) = beta;
            f_draws.slice(iter - burn_iterations + 1) = f;
        }
    }
    Rprintf("\r100.000 %% complete\n");
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("beta", beta_draws),
                                           Rcpp::Named("f", f_draws));
    return result;
}

