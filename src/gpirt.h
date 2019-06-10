#include <RcppArmadillo.h>
#include <truncnorm.h>
#include <mvnorm.h>

Rcpp::List gpirtMCMC(const arma::imat& y, const int sample_iterations,
                     const int burn_iterations);

// Function to draw f
arma::mat draw_f(const arma::mat& f, const arma::imat& y, const arma::mat& S);

// Function to draw theta
arma::vec draw_theta(const arma::imat& y,
                     const double ell, const double sf,
                     const arma::mat& f,
                     const arma::vec& theta,
                     const arma::vec& theta_star,
                     const arma::vec& theta_prior);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, double sf, double ell);

// Likelihood function
double ll(const arma::vec& f, const arma::ivec& y);

