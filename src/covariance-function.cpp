#include <RcppArmadillo.h>

arma::mat K(const arma::vec& x1, const arma::vec& x2,
            const double sf_sq, const double ell_sq_reciprocal) {
    arma::uword n = x1.n_elem;
    arma::uword m = x2.n_elem;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double d = x1[i] - x2[j];
            result(i, j) = sf_sq * std::exp(-0.5 * d * d * ell_sq_reciprocal);
        }
    }
    return result;
}

