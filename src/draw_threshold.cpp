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
arma::vec ess_threshold(const arma::vec& delta, const arma::mat& f,
                const arma::mat& y, const arma::mat& mu) {
    arma::uword C = delta.n_elem + 1;
    arma::uword n = mu.n_rows;
    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec v(C-1, arma::fill::ones);
    v *= 0.25;
    arma::mat S = arma::diagmat(v);
    arma::mat cholS = arma::chol(S, "lower");
    arma::vec nu = rmvnorm(cholS);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0, 1.0);
    double log_y = std::log(u);
    arma::vec thresholds = delta_to_threshold(delta);
    for (arma::uword i = 0; i < n; i++)
    {
        log_y += ll_bar(f.row(i).t(), y.row(i).t(), mu.row(i).t(), thresholds);
    }

    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for delta_prime out of the loop
    arma::vec delta_prime(C-1, arma::fill::zeros);
    int iter = 0;

    while ( reject ) {
        iter += 1;
        // Get f_prime given current epsilon
        delta_prime = delta * std::cos(epsilon) + nu * std::sin(epsilon);
        // If the log likelihood is over our threshold, accept
        double log_y_prime = 0;
        arma::vec thresholds_prime = delta_to_threshold(delta_prime);
        for (arma::uword i = 0; i < n; i++)
        {
            log_y_prime += ll_bar(f.row(i).t(), y.row(i).t(), mu.row(i).t(), thresholds_prime);
        }
        if ( log_y_prime > log_y ) {
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
    return delta_prime;
}

// Function to draw thresholds

arma::vec draw_threshold(const arma::vec& thresholds, const arma::mat& y,
                    const arma::mat& f, const arma::mat& mu){
    arma::vec delta = threshold_to_delta(thresholds);
    arma::vec delta_prime = ess_threshold(delta, f, y, mu);
    return delta_to_threshold(delta_prime);
}
