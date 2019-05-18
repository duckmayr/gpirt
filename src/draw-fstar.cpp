#include "gpirt.h"

arma::mat draw_fstar(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& S00,
                     const arma::mat& S11) {
    int m = f.n_cols;
    int N = theta_star.n_elem;
    arma::mat result(N, m);
    arma::vec fj(N);
    arma::vec fmu(N);
    arma::mat S01  = K(theta,      theta_star, 1.0, 1.0);
    // We use S10 * S00.i() in each loop iteration, so we store it now
    arma::mat S10_S00i = S01.t() * S00.i();
    arma::mat S = S11 - ( S10_S00i * S01 );
    S.diag() += 0.001;
    for ( arma::uword j = 0; j < m; ++j ) {
        fj = f.col(j);
        fmu = S10_S00i * fj;
        result.col(j) = rmvnorm(1, fmu, S).t();
    }
    return result;
}

