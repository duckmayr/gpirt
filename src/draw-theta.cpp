#include "gpirt.h"

arma::vec draw_theta(const int n, const arma::vec& theta_star,
                     const arma::mat& y, const arma::vec& theta0_prior,
                     const arma::vec& thetai_prior, const arma::mat& fstar) {
    int N = theta_star.n_elem;
    int m = fstar.n_cols;
    arma::vec result(n);
    arma::vec responses(m);
    arma::vec fk(m);
    arma::vec P(N);
    for ( arma::uword i = 0; i < n; ++i ) {
        // For each respondent,
        responses = y.row(i).t();
        for ( arma::uword k = 0; k < N; ++k ) {
            // For each value in theta_star,
            // get the log prior + the log likelihood
            if ( i == 0 ) {
                P[k] = theta0_prior[k];
            }
            else {
                P[k] = thetai_prior[k];
            }
            fk = fstar.row(k).t();
            P[k] += ll(fk, responses);
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
