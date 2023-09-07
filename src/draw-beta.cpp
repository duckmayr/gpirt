#include "gpirt.h"
#include "mvnormal.h"

// arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
//                     const arma::cube& y, const arma::cube& f,
//                     const arma::mat& prior_means, const arma::mat& prior_sds,
//                     const arma::mat& step_sizes, const arma::vec& thresholds) {
//     // Bookkeeping variables
//     arma::uword p = beta.n_rows; // # of mean function variables (2 for now)
//     arma::uword m = beta.n_cols; // # of response functions we are learning
//     arma::uword horizon = beta.n_slices; // # of time periods

//     // Setup result object
//     arma::cube result(p, m, horizon);

//     // Update coefficients (MH steps) one at a time by rule then by variable
//     for (arma::uword h = 0; h < horizon; h++){
//         for ( arma::uword j = 0; j < m; ++j ) {
//             arma::vec responses = y.slice(h).col(j);
//             arma::vec cv = beta.slice(h).col(j);
//             arma::vec pv(cv);
//             arma::vec rho = f.slice(h).col(j);
//             for ( arma::uword k = 0; k < p; ++k ) {
//                 pv[k]             = R::rnorm(cv[k], step_sizes(k, j));
//                 double prior_mean = prior_means(k, j);
//                 double prior_sd   = prior_sds(k, j);
//                 double pv_prior   = R::dnorm(pv[k], prior_mean, prior_sd, 1);
//                 double cv_prior   = R::dnorm(cv[k], prior_mean, prior_sd, 1);
//                 double pv_ll      = ll_bar(rho, responses, X.slice(h) * pv, thresholds);
//                 double cv_ll      = ll_bar(rho, responses, X.slice(h) * cv, thresholds);
//                 double r          = pv_prior + pv_ll - cv_prior - cv_ll;
//                 if ( std::log(R::runif(0.0,1.0)) < r ) {
//                     cv[k] = pv[k];
//                 }
//                 else {
//                     pv[k] = cv[k];
//                 }
//             }
//             result.slice(h).col(j) = cv;
//         }

//     }
    
//     return result;
// }

inline arma::vec draw_beta_ess(const arma::vec& beta, 
                               const arma::vec& f, 
                               const arma::vec& y, 
                               const arma::mat& cholS,
                               const arma::mat& X, 
                               const arma::vec& thresholds) {
    arma::uword n = 2;
    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec nu = rmvnorm(cholS);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0, 1.0);
    // double u = arma::randu(1);
    double log_y = ll_bar(f, y, X*beta, thresholds) + std::log(u);
    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for f_prime out of the loop
    arma::vec beta_prime(n);
    while ( reject ) {
        // Get f_prime given current epsilon
        beta_prime = beta * std::cos(epsilon) + nu * std::sin(epsilon);
        // If the log likelihood is over our threshold, accept
        if ( ll_bar(f, y, X*beta_prime, thresholds) > log_y ) {
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
    return beta_prime;
}


arma::cube draw_beta(arma::cube& beta, const arma::cube& X,
                    const arma::cube& y, const arma::cube& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::cube& thresholds) {
    // Bookkeeping variables
    arma::uword p = beta.n_rows; // # of mean function variables (2 for now)
    arma::uword m = beta.n_cols; // # of response functions we are learning
    arma::uword horizon = beta.n_slices; // # of time periods

    // Setup result object
    arma::cube result(p, m, horizon);

    // Update coefficients (ess) one at a time
    for (arma::uword h = 0; h < horizon; h++){
        for ( arma::uword j = 0; j < m; ++j ) {
            arma::mat cholS(3,3, arma::fill::zeros);
            cholS.diag() = prior_sds.col(j);
            cholS = arma::powmat(cholS,2);
            cholS.diag() += 1e-6;
            cholS = arma::chol(cholS, "lower");
            result.slice(h).col(j) = draw_beta_ess(beta.slice(h).col(j),\
            f.slice(h).col(j), y.slice(h).col(j), cholS, X.slice(h), thresholds.slice(h).row(j).t());
        }

    }
    
    return result;
}

