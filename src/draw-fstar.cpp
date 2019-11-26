#include "gpirt.h"

arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& S00,
                     const double sf, const double ell,
                     const arma::mat& mu, const arma::mat& mu_star) {
    int n = f.n_rows;
    int m = f.n_cols;
    int N = theta_star.n_elem;
    arma::mat result(N, m);
    arma::vec fj(n);
    arma::mat S00i = S00.i();
    arma::mat S01(n, 1);
    arma::mat S10_S00i(1, n);
    arma::vec th_star(1);
    for ( arma::uword i = 0; i < N; ++i ) {
        th_star[0] = theta_star[i];
        S01        = K(theta, th_star, sf, ell);
        S10_S00i   = S01.t() * S00i;
        double S   = (sf * sf) - arma::as_scalar(S10_S00i * S01);
        for ( arma::uword j = 0; j < m; ++j ) {
            fj = f.col(j) - mu.col(j);
            double fmu = mu_star(i, j) + arma::as_scalar(S10_S00i * fj);
            result(i, j) = R::rnorm(fmu, S);
        }
    }
    return result;
}

