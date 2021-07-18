#include "gpirt.h"

arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& S00,
                     const arma::mat& mu, const arma::mat& mu_star) {
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    arma::vec fj(n);
    arma::mat S01(n, 1);
    arma::mat S10   = K(theta_star, theta);
    arma::mat Ks_Ki = S10 * S00.i();
    arma::vec V     = arma::sum(Ks_Ki % S10, 1);
    arma::vec draw_mean(N);
    arma::mat S10_S00i(1, n);
    arma::vec th_star(1);
    for ( arma::uword j = 0; j < m; ++j ) {
        // draw_mean = Ks_Ki * f.col(j);
        draw_mean = mu_star.col(j) + Ks_Ki * f.col(j);
        for ( arma::uword i = 0; i < N; ++i ) {
            result(i, j) = R::rnorm(draw_mean[i], 1.0 - V[i]);
        }
    }
    return result;
}
