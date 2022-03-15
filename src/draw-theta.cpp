#include "gpirt.h"

arma::vec draw_theta(const arma::vec& theta_star,
                     const arma::mat& y, const arma::vec& theta_prior,
                     const arma::mat& fstar, const arma::mat& mu_star,
                     const arma::vec& thresholds) {
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
            P[k] = theta_prior[k] + ll_bar(fstar.row(k).t(),
                     responses, mu_star.row(k).t(), thresholds);
        }
        // Exponeniate, cumsum, then scale to [0, 1] for the "CDF"
        P = arma::exp(P);
        P = arma::cumsum(P);
        P = (P - P.min()) / (P.max() - P.min());
        // Then (sort of) inverse sample
        double u = R::runif(0.0, 1.0);
        result[i] = theta_star[arma::sum(P<=u)];
    }
    return result;
}
