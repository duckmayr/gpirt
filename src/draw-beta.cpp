#include "gpirt.h"

arma::mat draw_beta(const arma::mat& beta, const arma::mat& X,
                    const arma::mat& y, const arma::mat& f,
                    const arma::mat& prior_means, const arma::mat& prior_sds,
                    const arma::mat& step_sizes, const arma::vec& thresholds) {
    
    // Bookkeeping variables
    arma::uword p = beta.n_rows; // # of mean function variables (2 for now)
    arma::uword m = beta.n_cols; // # of response functions we are learning

    // Setup result object
    arma::mat result(p, m);

    // Update coefficients (MH steps) one at a time by rule then by variable
    for ( arma::uword j = 0; j < m; ++j ) {
        arma::vec responses = y.col(j);
        arma::vec cv = beta.col(j);
        arma::vec pv(cv);
        arma::vec rho = f.col(j);
        for ( arma::uword k = 0; k < p; ++k ) {
            pv[k]             = R::rnorm(cv[k], step_sizes(k, j));
            double prior_mean = prior_means(k, j);
            double prior_sd   = prior_sds(k, j);
            double pv_prior   = R::dnorm(pv[k], prior_mean, prior_sd, 1);
            double cv_prior   = R::dnorm(cv[k], prior_mean, prior_sd, 1);
            double pv_ll      = ll_bar(rho, responses, X * pv, thresholds);
            double cv_ll      = ll_bar(rho, responses, X * cv, thresholds);
            double r          = pv_prior + pv_ll - cv_prior - cv_ll;
            if ( std::log(R::runif(0.0, 1.0)) < r ) {
                cv[k] = pv[k];
            }
            else {
                pv[k] = cv[k];
            }
        }
        result.col(j) = cv;
    }

    return result;
}
