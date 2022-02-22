#include <RcppArmadillo.h>

// Function to draw f
arma::mat draw_f(const arma::mat& f, const arma::mat& y, const arma::mat& cholS,
                 const arma::mat& mu, const arma::mat& thresholds);

// Function to draw fstar
arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& L,
                     const arma::mat& mu_star);

// Function to draw theta
arma::vec draw_theta(const arma::vec& theta_star,
                     const arma::mat& y, const arma::vec& theta_prior,
                     const arma::mat& fstar, const arma::mat& mu_star,
                     const arma::mat& thresholds);

// Function to draw beta
arma::mat draw_beta(const arma::mat& beta, const arma::mat& X,
                    const arma::mat& y, const arma::mat& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::mat& proposal_sds, const arma::mat& thresholds);

// Function to draw thresholds
arma::mat draw_threshold(const arma::mat& thresholds, const arma::mat& y,
                    const arma::mat& f, const arma::mat& mu);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2);

// Likelihood function
// double ll(const arma::vec& f, const arma::vec& y);
// double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu);

// Likelihood function for ordinal regression
double ll(const arma::vec& f, const arma::vec& y, const arma::mat& thresholds);
double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds);

// convertion between thresholds and delta thresholds
arma::vec delta_to_threshold(const arma::vec& deltas);
arma::vec threshold_to_delta(const arma::vec& thresholds);
