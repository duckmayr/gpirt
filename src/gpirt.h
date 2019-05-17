#include <RcppArmadillo.h>
#include <mvnorm.h>

Rcpp::List gpirtMCMC(const arma::mat& y, const int sample_iterations,
                     const int burn_iterations);

// Function to draw f
arma::vec draw_f(const arma::vec& f, const arma::vec& y, const arma::mat& S);

// Function to draw fstar
arma::vec draw_fstar(const arma::vec& f, const arma::vec& theta,
                     const arma::vec& theta_star);

// Function to draw theta
arma::vec draw_theta(const arma::vec& theta_star, const arma::vec& fstar);

// Covariance function
arma::mat K(const arma::vec& x1, const arma::vec& x2, const double sf,
            const double ell);

// Likelihood function
double ll(const arma::vec& f, const arma::vec& y);

