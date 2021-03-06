#include <RcppArmadillo.h>

// LOG LIKELIHOOD FUNCTION
/* (Explainer using LaTeX if you want to copy/paste/compile to view pretty)
 * The likelihood is $L = \prod\frac{1}{1+\exp(-y_i f_i)}$, so
 * \begin{align*}
 *     \log(L) &= \sum\left[\log\left(\frac{1}{1+\exp(-y_if_i)}\right)\right] \\
 *             &= \sum\left[\log(1) - \log(1 + \exp(-y_i f_i))\right] \\
 *             &= -\sum\left[\log(1 + \exp(-y_i f_i))\right]
 * \end{align*}
 */
double ll(const arma::vec& f, const arma::vec& y) {
    int n = f.n_elem;
    double result = 0.0;
    for ( arma::uword i = 0; i < n; ++i ) {
        if ( std::isnan(y[i]) ) {
            continue;
        }
        double a = y[i] * f[i];
        result -= std::log(1 + std::exp(-a));
    }
    return result;
}

double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu) {
    int n = f.n_elem;
    double result = 0.0;
    arma::vec g = f + mu;
    for ( arma::uword i = 0; i < n; ++i ) {
        if ( std::isnan(y[i]) ) {
            continue;
        }
        double a = y[i] * g[i];
        result -= std::log(1 + std::exp(-a));
    }
    return result;
}
