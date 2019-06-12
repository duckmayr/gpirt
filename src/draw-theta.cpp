#include "gpirt.h"

arma::vec draw_theta(const arma::imat& y,
                     const double ell_sq_reciprocal, const double sf_sq,
                     const arma::mat& f,
                     const arma::vec& theta,
                     const arma::vec& theta_star,
                     const arma::vec& L_prior,
                     const arma::vec& R_prior,
                     const arma::ivec& party) {
    arma::uword N = theta_star.n_elem;
    arma::uword m = y.n_cols;
    arma::uword n = y.n_rows;
    arma::vec result(n);
    arma::ivec responses(m);
    arma::vec P(N);
    arma::mat S01(n-1, 1);
    arma::mat S10_S00i(1, n-1);
    arma::vec th_star(1);
    for ( arma::uword i = 0; i < n; ++i ) {
        // For each respondent,
        responses = y.row(i).t();
        arma::vec theta_not_i = theta;
        theta_not_i.shed_row(i);
        arma::mat f_not_i = f;
        f_not_i.shed_row(i);
        arma::mat S00i = K(theta_not_i, theta_not_i, sf_sq, ell_sq_reciprocal);
        S00i.diag() += 0.000001;
        S00i = inv_sympd(S00i);
        for ( arma::uword k = 0; k < N; ++k ) {
            // For each value in theta_star,
            // get the log prior + the log likelihood
            th_star[0] = theta_star[k];
            S01        = K(theta_not_i, th_star, sf_sq, ell_sq_reciprocal);
            S10_S00i   = S01.t() * S00i;
            double S   = sf_sq - arma::as_scalar(S10_S00i * S01);
            if ( party[i] ) {
                P[k] = R_prior[k];
            }
            else {
                P[k] = L_prior[k];
            }
            for ( arma::uword j = 0; j < m; ++j ) {
                if ( y(i, j) == INT_MIN ) {
                    continue;
                }
                arma::vec fj = f_not_i.col(j);
                double mu_j = arma::as_scalar(S10_S00i * fj);
                double mean = mu_j / (std::sqrt(1 + S));
                P[k] += R::pnorm(mean, 0.0, 1.0, y(i, j), 1);
            }
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

