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

arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::cube& L,
                      const arma::cube& mu_star,
                      const int constant_IRF) {
    arma::uword n = f.n_rows;
    arma::uword horizon = f.n_slices;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::cube results = arma::zeros<arma::cube>(N, m, horizon);
    
    if(constant_IRF==0){
        // draw fstar separately for non-constant IRF
        for ( arma::uword h = 0; h < horizon; ++h ){
            results.slice(h) = draw_fstar_(f.slice(h), theta.col(h), \
                                theta_star, L.slice(h), mu_star.slice(h));
        }
    }
    else{
        // draw fstar for constant IRF using all horizon theta/f
        // reshape f/theta
        arma::mat f_constant(n*horizon, m);
        arma::vec theta_constant(n*horizon);
        for (arma::uword h = 0; h < horizon; h++){
            theta_constant.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for ( arma::uword j = 0; j < m; ++j ) {
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
            }
        }
        // Set up X
        arma::mat X_constant(n*horizon, 2);
        X_constant.col(0) = arma::ones<arma::vec>(n*horizon);
        X_constant.col(1) = theta_constant;
        // Set up S
        arma::mat S_constant = arma::zeros<arma::mat>(n*horizon, n*horizon);
        S_constant = K(theta_constant, theta_constant);
        S_constant.diag() += 1e-6;
        // Set up L
        arma::vec beta_prior_sds = 0.5*arma::ones<arma::vec>(2);
        arma::mat L_constant(n*horizon, n*horizon);
        L_constant = arma::chol(S_constant + \
                        X_constant*arma::diagmat(square(beta_prior_sds))* \
                        X_constant.t(), "lower");
        arma::mat f_star(N, m);
        f_star = draw_fstar_(f_constant, theta_constant, \
                                theta_star, L_constant, mu_star.slice(0));

        // store the same fstar in the result object for all horizons
        for ( arma::uword h = 0; h < horizon; ++h ){
            results.slice(h) = f_star;
        }
    }
    
    return results;
}