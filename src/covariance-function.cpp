#include <RcppArmadillo.h>

arma::mat K(const arma::vec& x1, const arma::vec& x2, const double sf,
            const double ell) {
    arma::uword n = x1.n_elem;
    arma::uword m = x2.n_elem;
    arma::mat result(n, m);
    sf = sf * sf;
    ell = 1 / (ell * ell);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double diff = x1[i] - x2[j];
            result(i, j) = std::exp(0.5 * sf * -(diff * diff) * ell);
        }
    }
    return result;
}

