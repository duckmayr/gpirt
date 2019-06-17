#include <RcppArmadillo.h>
#include <truncnorm.h>
#include <mvnorm.h>

// Function to draw f
arma::mat draw_f(const arma::mat& f, const arma::mat& y, const arma::mat& S);

// Function to draw fstar
arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& S00,
                     const double sf, const double ell);

// Function to draw theta
arma::vec draw_theta(const int n, const arma::vec& theta_star,
                     const arma::mat& y, const arma::mat& theta_prior,
                     const arma::uvec& groups, const arma::mat& fstar);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, double sf, double ell);

// Likelihood function
double ll(const arma::vec& f, const arma::vec& y);

