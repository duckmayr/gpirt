#include <RcppArmadillo.h>

// Function to set seed state
void set_seed_state(Rcpp::NumericVector seed_state);
void set_seed(int seed);
Rcpp::NumericVector get_seed_state();

// Function to draw f
arma::cube draw_f(const arma::cube& f, const arma::mat& theta, const arma::cube& y, const arma::cube& cholS,
            const arma::mat& beta_prior_sds, const arma::cube& mu, const arma::cube& thresholds, const int constant_IRF);

// Function to draw fstar
arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
                      const arma::cube& L,
                      const arma::cube& mu_star,
                      const int constant_IRF);

// Function to draw theta
arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::mat& theta,
                     const arma::mat& theta_prior_sds,
                     const arma::cube& fstar, const arma::cube& mu_star,
                     const arma::cube& thresholds,
                     const double& os,
                     const double& ls, const std::string& KERNEL);

arma::vec draw_theta_0(const arma::vec& theta_star,
                     const arma::cube& y, const arma::mat& theta,
                     const arma::cube& fstar, const arma::cube& mu_star,
                     const arma::cube& thresholds,
                     const arma::vec& theta_prior_means,
                     const arma::vec& theta_prior_sds,
                     const arma::vec& theta_step_sizes);

// Function to draw beta
arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
                    const arma::cube& y, const arma::cube& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::cube& thresholds);

// Function to draw thresholds
arma::cube draw_threshold(const arma::cube& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu, const int constant_IRF);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, const arma::vec& beta_prior_sds);
arma::mat K_time(const arma::vec& x1, const arma::vec& x2,
                 const double& os, const double& ls,
                 const arma::vec& theta_prior_sds, const std::string& KERNEL);

// Likelihood function
// double ll(const arma::vec& f, const arma::vec& y);
// double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu);

// Likelihood function for ordinal regression
double ll(const arma::vec& f, const arma::vec& y, const arma::mat& thresholds);
double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds);

// convertion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);

// cholesky decomposition
arma::mat double_solve(const arma::mat& L, const arma::mat& X);
arma::mat compress_toeplitz(arma::mat& T);
arma::mat toep_cholesky_lower(arma::mat& T);
