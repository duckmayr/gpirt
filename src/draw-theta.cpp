#include "gpirt.h"

arma::vec draw_theta(const arma::vec& theta_star,
                     const arma::mat& y, const arma::vec& theta_prior,
                     const arma::mat& fstar, const arma::mat& mu_star) {
    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::vec result(n);
    arma::vec responses(m);
    arma::vec P(N);
    for ( arma::uword i = 0; i < n; ++i ) {
        // For each respondent, extract their responses
        responses = y.row(i).t();
        for ( arma::uword k = 0; k < N; ++k ) {
            // Then for each value in theta_star,
            // get the log prior + the log likelihood
            P[k] = theta_prior[k] + ll(fstar.row(k).t(), responses);
        }
        // Exponeniate, cumsum, then scale to [0, 1] for the "CDF"
        P = arma::exp(P);
        P = arma::cumsum(P);
        double max_p = P.max();
        double min_p = P.min();
        P = (P - min_p) / (max_p - min_p);
        // Then (sort of) inverse sample
        double u = R::runif(0.0, 1.0);
        result[i] = theta_star[N];
        for ( arma::uword k = 0; k < N; ++k ) {
            if ( P[k] > u ) {
                result[i] = theta_star[k];
                break;
            }
        }
    }
    return result;
}
