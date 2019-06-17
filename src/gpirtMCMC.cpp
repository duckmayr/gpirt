#include "gpirt.h"

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::mat& y, arma::vec theta,
                     const int sample_iterations, const int burn_iterations,
                     const arma::vec& means, const arma::uvec& groups,
                     const double sf, const double ell) {
    int n = y.n_rows;
    int m = y.n_cols;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta and f
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::mat S = K(theta, theta, sf, ell);
    S.diag() += 0.001;
    arma::mat f = rmvnorm(m, mean_zeros, S).t();
    // Setup grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    int N = theta_star.n_elem;
    arma::mat f_star(N, m);
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
    arma::cube f_draws(n, m, sample_iterations + 1);
    // Store initial values
    theta_draws.row(0) = theta.t();
    f_draws.slice(0)   = f;
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
        f = draw_f(f, y, S);
        f_star = draw_fstar(f, theta, theta_star, S, sf, ell);
        theta = draw_theta(n, theta_star, y, theta_prior, groups, f_star);
        S = K(theta, theta, sf, ell);
        S.diag() += 0.001;
        // Store draws
        if ( iter >= burn_iterations ) {
            theta_draws.row(iter - burn_iterations + 1) = theta.t();
            f_draws.slice(iter - burn_iterations + 1) = f;
        }
    }
    Rprintf("\r100.000 %% complete\n");
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("f", f_draws));
    return result;
}

