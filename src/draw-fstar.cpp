#include "gpirt.h"
#include "mvnormal.h"

arma::mat double_solve(const arma::mat& L, const arma::mat& X) {
    using arma::trimatl;
    using arma::trimatu;
    using arma::solve;
    return solve(trimatu(L.t()), solve(trimatl(L), X));
}

inline arma::mat draw_fstar_(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& L,
                     const arma::mat& mu_star) {
    // draw fstar for one horizon
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    arma::mat kstar  = K(theta, theta_star);
    arma::mat kstarT = kstar.t();
    arma::mat tmp = arma::solve(arma::trimatl(L), kstar);
    arma::mat K_post = K(theta_star, theta_star) - tmp.t() * tmp;
    K_post.diag() += 1e-6;
    arma::mat L_post = arma::chol(K_post, "lower");
    arma::vec alpha(n);
    arma::vec draw_mean(N);
    for ( arma::uword j = 0; j < m; ++j ) {
        alpha = double_solve(L, f.col(j));
        draw_mean = kstarT * alpha + mu_star.col(j);
        result.col(j) = draw_mean + rmvnorm(L_post);
    }
    return result;
}

arma::cube draw_fstar(const arma::cube& f, const arma::mat& theta,
                     const arma::vec& theta_star, const arma::cube& L,
                     const arma::mat& mu_star) {
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::cube results(N, m, horizon);

    for ( arma::uword h = 0; h < horizon; ++h ){
        results.slice(h) = draw_fstar_(f.slice(h), theta.col(h), \
                                theta_star, L.slice(h), mu_star);
    }

    return results;
}