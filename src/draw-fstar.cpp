#include "gpirt.h"
#include "mvnormal.h"

arma::mat double_solve(const arma::mat& L, const arma::mat& X) {
    using arma::trimatl;
    using arma::trimatu;
    using arma::solve;
    return solve(trimatu(L.t()), solve(trimatl(L), X));
}

inline arma::mat draw_fstar_(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& beta_prior_sds,
                     const arma::mat& L, const arma::mat& mu_star) {
    // draw fstar for one horizon
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        arma::mat kstar  = K(theta, theta_star, beta_prior_sds.col(j));
        arma::mat kstarT = kstar.t();
        arma::mat tmp = arma::solve(arma::trimatl(L), kstar);
        arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(j)) - tmp.t() * tmp;
        K_post.diag() += 1e-6;
        arma::mat L_post = arma::chol(K_post, "lower");
        arma::vec alpha(n);
        arma::vec draw_mean(N);
        alpha = double_solve(L, f.col(j));
        draw_mean = kstarT * alpha + mu_star.col(j);
        // draw_mean = kstarT * alpha;
        result.col(j) = draw_mean + rmvnorm(L_post);
    }
    return result;
}

arma::cube draw_fstar(const arma::cube& f, 
                      const arma::mat& theta,
                      const arma::vec& theta_star, 
                      const arma::mat& beta_prior_sds,
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
                                theta_star, beta_prior_sds, L.slice(h), mu_star.slice(h));
        }
    }
    else{
        // draw fstar for constant IRF using all horizon theta/f
        // reshape f/theta
        arma::mat f_constant_all(n*horizon, m);
        arma::vec theta_constant_all(n*horizon);
        for (arma::uword h = 0; h < horizon; h++){
            theta_constant_all.subvec(h*n, (h+1)*n-1) = theta.col(h);
            for ( arma::uword j = 0; j < m; ++j ) {
                f_constant_all.col(j).subvec(h*n, (h+1)*n-1) = f.slice(h).col(j);
            }
        }
        // implement naive 100 inducing points
        // evenly spread inducing locations in the range of current theta
        // use knn for inducing values 
        int n_induced_points = 100;
        arma::mat f_constant(n_induced_points, m);
        arma::vec theta_constant(f_constant.n_rows);
        theta_constant = arma::linspace(theta.min(), theta.max(), n_induced_points);
        for (arma::uword j = 0; j < m; ++j)
        {
            // sort theta_constant_all/f_constant_all
            // arma::uvec indices = sort_index(f_constant_all.col(h));
            // f_constant_all.col(h) = f_constant_all.col(h)[indices];
            // theta_constant_all.col(h) = theta_constant_all.col(h)[indices];
            // interpolate induced values
            arma::vec points;
            arma::interp1(theta_constant_all, f_constant_all.col(j).t(),
                    theta_constant, points, "linear");
            f_constant.col(j) = points;
        }
        
        // Set up X
        arma::mat X_constant(f_constant.n_rows, 3);
        X_constant.col(0) = arma::ones<arma::vec>(f_constant.n_rows);
        X_constant.col(1) = theta_constant;
        X_constant.col(2) = arma::pow(theta_constant,2);
        // Set up S
        arma::mat S_constant = arma::zeros<arma::mat>(f_constant.n_rows, f_constant.n_rows);
        S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;
        // Set up L
        // arma::vec beta_prior_sds = 3*arma::ones<arma::vec>(3);
        arma::mat L_constant(f_constant.n_rows, f_constant.n_rows);
        // L_constant = arma::chol(S_constant + \
        //                 X_constant*arma::diagmat(square(beta_prior_sds))* \
        //                 X_constant.t(), "lower");
        L_constant = arma::chol(S_constant, "lower");
        arma::mat f_star(N, m);
        f_star = draw_fstar_(f_constant, theta_constant, \
                                theta_star,beta_prior_sds, L_constant, mu_star.slice(0));

        // store the same fstar in the result object for all horizons
        for ( arma::uword h = 0; h < horizon; ++h ){
            results.slice(h) = f_star;
        }
    }
    
    return results;
}