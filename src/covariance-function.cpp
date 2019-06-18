#include <RcppArmadillo.h>

arma::mat K(const arma::vec& x1, const arma::vec& x2, double sf, double ell) {
    arma::uword n = x1.n_elem;
    arma::uword m = x2.n_elem;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double diff = x1[i] - x2[j];
            result(i, j) = sf * std::exp(-0.5 * diff * diff * ell);
        }
    }
    return result;
}

