#include "gpirt.h"

// ELLIPTICAL SLICE SAMPLER
/*
 * For more details, the elliptical slice sampling paper can be found at
 * http://proceedings.mlr.press/v9/murray10a/murray10a.pdf
 * The following function is a fairly straight forward translation of their
 * algorithm on page 543 to C++.
 *
 * The basic trick is replacing f ~ N(0, Σ) with f = ν_0 sin ε + ν_1 cos ε,
 * ε ∼ Uniform[0, 2π], ν_0 ~ N(0, Σ), ν_1 ~ N(0, Σ), so that the marginal
 * distribution over f is still N(0, Σ), and for a draw ν and step size ε,
 * f' = ν sin ε + f cos ε draws an ellipse through ν and the current state of f
 * For a draw of ν, we can then adjust the step size to find an acceptable draw
 *
 * f is the initial values for f,
 * y is the response, and
 * S is the covariance matrix,
 */
arma::vec ess(const arma::vec& f, const arma::vec& y, const arma::mat& S,
              const arma::mat& mu) {
    arma::uword n = f.n_elem;
    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma. rmvnorm() will return
    // a 1 x n matrix that we transpose to an n x 1 vector
    arma::mat tmp = rmvnorm(1, arma::zeros<arma::colvec>(n), S);
    arma::vec nu = tmp.t();
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0, 1.0);
    double log_y = ll_bar(f, y, mu) + std::log(u);
    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for f_prime out of the loop
    arma::vec f_prime(n);
    int iter = 0;
    while ( reject ) {
        iter += 1;
        // Get f_prime given current epsilon
        f_prime = f * std::cos(epsilon) + nu * std::sin(epsilon);
        // If the log likelihood is over our threshold, accept
        if ( ll_bar(f_prime, y, mu) > log_y ) {
            reject = false;
        }
        // otw, adjust our proposal band & draw a new epsilon, then repeat
        else {
            if ( epsilon < 0.0 ) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return f_prime;
}

// Function to draw f

arma::mat draw_f(const arma::mat& f, const arma::mat& y, const arma::mat& S,
                 const arma::mat& mu) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j) {
        result.col(j) = ess(f.col(j), y.col(j), S, mu.col(j));
    }
    return result;
}
