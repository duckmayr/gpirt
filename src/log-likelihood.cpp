#include <RcppArmadillo.h>
#include "gpirt.h"

// LOG LIKELIHOOD FUNCTION
/* (Explainer using LaTeX if you want to copy/paste/compile to view pretty)
 * The likelihood is $L = \prod\frac{1}{1+\exp(-y_i f_i)}$, so
 * \begin{align*}
 *     \log(L) &= \sum\left[\log\left(\frac{1}{1+\exp(-y_if_i)}\right)\right] \\
 *             &= \sum\left[\log(1) - \log(1 + \exp(-y_i f_i))\right] \\
 *             &= -\sum\left[\log(1 + \exp(-y_i f_i))\right]
 * \end{align*}
 */
// double ll(const arma::vec& f, const arma::vec& y, const arma::vec& threshold) {
//     int n = f.n_elem;
//     double result = 0.0;
//     for ( arma::uword i = 0; i < n; ++i ) {
//         if ( std::isnan(y[i]) ) {
//             continue;
//         }
//         double a = y[i] * f[i];
//         result -= std::log(1 + std::exp(-a));
//     }
//     return result;
// }

// double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& threshold) {
//     int n = f.n_elem;
//     double result = 0.0;
//     arma::vec g = f + mu;
//     for ( arma::uword i = 0; i < n; ++i ) {
//         if ( std::isnan(y[i]) ) {
//             continue;
//         }
//         double a = y[i] * g[i];
//         result -= std::log(1 + std::exp(-a));
//     }
//     return result;
// }

// LOG LIKELIHOOD FUNCTION
/* (Explainer using LaTeX if you want to copy/paste/compile to view pretty)
 * The likelihood is $L(y=c|f(x)) = \Phi(z_1)-\Phi(z_2)$, where
 * \begin{align*} z_1 &= b_{c-1} - f(x) \\
 * z_2 &= b_{c} - f(x) \\
 * Phi(z) &= \int_{-\infty}^z N(x;0,1) dx \end{align*}
 */

double ll(const arma::vec& f, const arma::vec& y, const arma::mat& thresholds) {
    // each respondent for all items
    arma::uword m = f.n_elem;
    double result = 0.0;
    for ( arma::uword j = 0; j < m; ++j ) {
        if ( !std::isnan(y[j]) ) {
            int c = int(y[j]);
            double z1 = thresholds(j % thresholds.n_rows, c-1) - f[j];
            double z2 = thresholds(j % thresholds.n_rows, c) - f[j];
            result += std::log(R::pnorm(z2, 0, 1, 1, 0)-R::pnorm(z1, 0, 1, 1, 0)+1e-6);
        }
    }
    return result;
}

double ll_bar(const arma::vec& f, const arma::vec& y, const arma::vec& mu, const arma::vec& thresholds) {
    arma::uword n = f.n_elem;
    double result = 0.0;
    arma::vec g = f + mu;
    for ( arma::uword i = 0; i < n; ++i ) {
        if ( std::isnan(y[i]) ) {
            continue;
        }
        int c = int(y[i]);
        double z1 = thresholds[c-1] - g[i];
        double z2 = thresholds[c] - g[i];
        result += std::log(R::pnorm(z2, 0, 1, 1, 0)-R::pnorm(z1, 0, 1, 1, 0)+1e-6);
    }
    return result;
}


arma::vec delta_to_threshold(const arma::vec& deltas){
    // delta = (b_1, log(b_2-b_1), ..., log(b_{C-1}-b_{C-2}))
    // b_0 = -inf, b_C = +inf
    arma::uword C = deltas.n_elem + 1;
    arma::vec thresholds(C+1, arma::fill::zeros);
    thresholds[0] = -std::numeric_limits<double>::infinity();
    thresholds[1] = deltas[0];
    thresholds[C] = std::numeric_limits<double>::infinity();
    for (arma::uword i = 1; i < C-1; i++)
    {
        thresholds[i+1] = thresholds[i] + std::exp(deltas[i]);
    }
    return thresholds;
}

arma::vec threshold_to_delta(const arma::vec& thresholds){
    // delta = (b_1, log(b_2-b_1), ..., log(b_{C-1}-b_{C-2}))
    // b_0 = -inf, b_C = +inf
    arma::uword C = thresholds.n_elem - 1;
    arma::vec deltas(C-1, arma::fill::zeros);
    deltas[0] = thresholds[1];
    for (arma::uword i = 1; i < C-1; i++)
    {
        deltas[i] = std::log(thresholds[i+1]-thresholds[i]);
    }
    return deltas;
}
