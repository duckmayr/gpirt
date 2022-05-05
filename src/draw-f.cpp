#include "gpirt.h"
#include "mvnormal.h"

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
arma::vec ess(const arma::vec& f, const arma::vec& y, const arma::mat& cholS,
              const arma::vec& mu, const arma::vec& thresholds) {
    arma::uword n = f.n_elem;
    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec nu = rmvnorm(cholS);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    // double u = R::runif(0.0, 1.0);
    double u = arma::randu(1);
    double log_y = ll_bar(f, y, mu, thresholds) + std::log(u);
    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    // double epsilon = R::runif(epsilon_min, epsilon_max);
    double epsilon = (epsilon_max-epsilon_min)*arma::randu(1) + epsilon_min;
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for f_prime out of the loop
    arma::vec f_prime(n);
    int iter = 0;
    while ( reject ) {
        iter += 1;
        // Get f_prime given current epsilon
        f_prime = f * std::cos(epsilon) + nu * std::sin(epsilon);
        // If the log likelihood is over our threshold, accept
        if ( ll_bar(f_prime, y, mu, thresholds) > log_y ) {
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
            // epsilon = R::runif(epsilon_min, epsilon_max);
            epsilon = (epsilon_max-epsilon_min)*arma::randu(1) + epsilon_min;
        }
    }
    return f_prime;
}

// Function to draw f

inline arma::mat draw_f_(const arma::mat& f, const arma::mat& y, const arma::mat& cholS,
                 const arma::mat& mu, const arma::vec& thresholds) {
    // draw f for one horizon
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j) {
        result.col(j) = ess(f.col(j), y.col(j), cholS, mu.col(j), thresholds);
    }
    return result;
}

arma::cube draw_f(const arma::cube& f, const arma::cube& y, const arma::cube& cholS,
                 const arma::cube& mu, const arma::vec& thresholds) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword horizon = f.n_slices;
    arma::cube result(n, m, horizon);
    for ( arma::uword h = 0; h < horizon; ++h){
        result.slice(h) = draw_f_(f.slice(h), y.slice(h), cholS.slice(h),\
                        mu.slice(h), thresholds);
    }
    return result;
}
