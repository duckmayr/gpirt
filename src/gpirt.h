#include <RcppArmadillo.h>
#include <truncnorm.h>
#include <mvnorm.h>

Rcpp::List gpirtMCMC(const arma::mat& y, const int sample_iterations,
                     const int burn_iterations);

// Function to draw f
arma::mat draw_f(const arma::mat& f, const arma::mat& y, const arma::mat& S);

// Function to draw fstar
arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& S00,
                     const arma::mat& S11);

// Function to draw theta
arma::vec draw_theta(const int n, const arma::vec& theta_star,
                     const arma::mat& y, const arma::vec& theta0_prior,
                     const arma::vec& thetai_prior, const arma::mat& fstar);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, double sf, double ell);

// Likelihood function
double ll(const arma::vec& f, const arma::vec& y);

