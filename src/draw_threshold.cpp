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
arma::vec ess_threshold(const arma::vec& delta, const arma::cube& f,
                const arma::cube& y, const arma::cube& mu) {
    arma::uword C = delta.n_elem + 1;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec v(C-1, arma::fill::ones);
    arma::mat S = arma::diagmat(v);
    arma::mat cholS = arma::chol(S, "lower");
    arma::vec nu = rmvnorm(cholS);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0,1.0);
    double log_y = std::log(u);
    arma::vec thresholds = delta_to_threshold(delta);
    for (arma::uword h = 0; h < horizon; h++)
    {
        for (arma::uword i = 0; i < m; i++){
            log_y += ll_bar(f.slice(h).col(i), y.slice(h).col(i), 
                            mu.slice(h).col(i), thresholds);
        }
    }

    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    // double epsilon = (epsilon_max-epsilon_min)*arma::randu(1) + epsilon_min;
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for delta_prime out of the loop
    arma::vec delta_prime(C-1, arma::fill::zeros);

    while ( reject ) {
        // Get f_prime given current epsilon
        delta_prime = delta * std::cos(epsilon) + nu * std::sin(epsilon);
        // If the log likelihood is over our threshold, accept
        double log_y_prime = 0;
        arma::vec thresholds_prime = delta_to_threshold(delta_prime);
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword i = 0; i < m; i++)
            {
                log_y_prime += ll_bar(f.slice(h).col(i), y.slice(h).col(i),
                                      mu.slice(h).col(i), thresholds_prime);
            }
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

arma::cube draw_threshold(const arma::cube& thresholds, const arma::cube& y,
                    const arma::cube& f, const arma::cube& mu, const int constant_IRF){
    arma::uword m = thresholds.n_rows;
    arma::uword C = thresholds.n_cols - 1;
    arma::uword horizon = thresholds.n_slices;
    arma::cube thresholds_prime(m, C+1, horizon, arma::fill::zeros);
    if(constant_IRF==1){
        for ( arma::uword j = 0; j < m; ++j ){
            arma::vec delta = threshold_to_delta(thresholds.slice(0).row(j).t());
            arma::vec delta_prime = ess_threshold(delta, f, y, mu);
            thresholds_prime.slice(0).row(j) = delta_to_threshold(delta_prime).t();
            for(arma::uword h = 1; h < horizon; ++h){
                thresholds_prime.slice(h).row(j) = thresholds_prime.slice(0).row(j);
            }
        }
    }
    else{
        for ( arma::uword h = 0; h < horizon; ++h){
            for ( arma::uword j = 0; j < m; ++j ){
                arma::vec delta = threshold_to_delta(thresholds.slice(h).row(j).t());
                arma::vec delta_prime = ess_threshold(delta, f, y, mu);
                thresholds_prime.slice(h).row(j) = delta_to_threshold(delta_prime).t();
            }
        }
        
    }
    
    return thresholds_prime;
}
