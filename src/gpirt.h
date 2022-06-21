#include <RcppArmadillo.h>

// Function to set seed state
void set_seed_state(Rcpp::NumericVector seed_state);
void set_seed(int seed);
Rcpp::NumericVector get_seed_state();

// Function to draw f
arma::cube draw_f(const arma::cube& f, const arma::cube& y, const arma::cube& cholS,
                 const arma::cube& mu, const arma::vec& thresholds);

// Function to draw fstar
arma::cube draw_fstar(const arma::cube& f, const arma::mat& theta,
                     const arma::vec& theta_star, const arma::cube& L,
                     const arma::mat& mu_star);

// Function to draw theta
arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::vec& theta_prior,
                     const arma::cube& fstar, const arma::mat& mu_star,
                     const arma::vec& thresholds,
                     const arma::mat& fix_theta_flag,
                     const arma::mat& fix_theta_value,
                     const double& os,
                     const double& ls);

// Function to draw beta
arma::mat draw_beta(const arma::mat& beta, const arma::mat& X,
                    const arma::mat& y, const arma::mat& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::mat& proposal_sds, const arma::vec& thresholds);

// Function to draw thresholds
arma::vec draw_threshold(const arma::vec& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2);
arma::mat K_time(const arma::vec& x1, const arma::vec& x2,
                 const double& os, const double& ls);

// Likelihood function
// double ll(const arma::vec& f, const arma::vec& y);
// double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu);

// Likelihood function for ordinal regression
double ll(const arma::vec& f, const arma::vec& y, const arma::vec& thresholds);
double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds);

// convertion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);

// cholesky decomposition
arma::mat double_solve(const arma::mat& L, const arma::mat& X);
