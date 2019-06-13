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
    arma::vec th_star(1);    // a test point (\theta_k^*)
    arma::mat S01(n-1, 1);   //      K(\theta_{-i}, \theta_k^*)
    arma::mat S10(1, n-1);   //      K(\theta_k^*, \theta_{-i})
    arma::mat S00(n-1, n-1); //      K(\theta_{-i}, \theta_{-i})
    arma::mat L(n-1, n-1);   // chol(K(\theta_{-i}, \theta_{-i}))
    arma::mat Lt(n-1, n-1);  // chol(K(\theta_{-i}, \theta_{-i}))'
    arma::mat Alpha(n-1, m); // stores the alpha calculations
    arma::vec fj(n-1);       // f(-i, j)
    arma::mat tmp(n-1, 1);   // L  \ fj
    arma::mat a(n-1, 1);     // L' \ tmp
    arma::vec v(n-1);        // L  \ S01
    for ( arma::uword i = 0; i < n; ++i ) {
        // For each respondent,
        responses = y.row(i).t();
        arma::vec theta_not_i = theta;
        theta_not_i.shed_row(i);
        arma::mat f_not_i = f;
        f_not_i.shed_row(i);
        S00 = K(theta_not_i, theta_not_i, sf_sq, ell_sq_reciprocal);
        S00.diag() += 0.000001;
        L  = arma::chol(S00, "lower");
        Lt = L.t();
        for ( arma::uword j = 0; j < m; ++j ) {
            fj = f_not_i.col(j);
            tmp = arma::solve(arma::trimatl(L), fj);
            a = arma::solve(arma::trimatu(Lt), tmp);
            Alpha.col(j) = a;
        }
        for ( arma::uword k = 0; k < N; ++k ) {
            // For each value in theta_star,
            // get the log prior + the log likelihood
            th_star[0] = theta_star[k];
            S01        = K(theta_not_i, th_star, sf_sq, ell_sq_reciprocal);
            S10        = S01.t();
            v          = arma::solve(L, S01);
            double S   = sf_sq - arma::dot(v, v);
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
                double mu_j = arma::as_scalar(S10 * Alpha.col(j));
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

