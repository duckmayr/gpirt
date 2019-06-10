#include <RcppArmadillo.h>

// LOG LIKELIHOOD FUNCTION
double ll(const arma::vec& f, const arma::ivec& y) {
    int n = f.n_elem;
    double result = 0.0;
    for ( arma::uword i = 0; i < n; ++i ) {
        if ( y[i] == INT_MIN ) {
            continue;
        }
        result += R::pnorm(f[i], 0.0, 1.0, y[i], 1);
    }
    return result;
}

