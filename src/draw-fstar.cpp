#include "gpirt.h"
#include "mvnormal.h"
#include <time.h>

// inline arma::mat chol_downdate(arma::mat& L, arma::mat& X){
// // downdate a cholesky decomposition
//     arma::uword n = L.n_rows;
//     arma::vec alpha = arma::zeros<arma::vec>(1,n);
//     arma::vec gamma = arma::zeros<arma::vec>(1,n);
//     arma::vec delta = arma::zeros<arma::vec>(1,n);
//     arma::vec beta = arma::ones<arma::vec>(1,n);
//     for (arma::uword j = 0; j < n; j++)
//     {
//         alpha = X.row(j) / L(j,j) ;
//         arma::vec beta2 = armd::sqrt(arma::square(beta) - arma::square(alpha));
//         gamma = alpha / (beta2 * beta);
//         delta = beta2 / beta ;
//         L(j,j) = delta(0) * L(j,j) ;
//         X.row(j) = alpha ;
//         beta = beta2 ;
//         if (j == n){ return L;}
//     }
//     return L;
// }

inline arma::mat chol_update(arma::mat& L, arma::vec& x, bool update=true){
// rank-one chol update for computing chol(LL^T+xx^T)
// update = 1 for update, update = 0 for downdate
    arma::uword n = L.n_rows;
    double r = 0.0;
    double c = 0.0;
    double s = 0.0;
    double SIGN = update?1.0:-1.0;

    for (arma::uword i = 0; i < n; i++){
        r = std::sqrt(std::pow(L(i, i),2) + SIGN * std::pow(x(i),2));
        c = r / L(i, i);
        s = x(i) / L(i, i);
        L(i, i) = r;
        if (i<n-1){
            L.col(i).subvec(i+1, n-1) = (L.col(i).subvec(i+1, n-1) + SIGN*s*x.subvec(i+1,n-1)) / c;
            x.subvec(i+1,n-1) = c*x.subvec(i+1,n-1) - s*L.col(i).subvec(i+1, n-1);
        }
    }
    return L;
}

arma::mat double_solve(const arma::mat& L, const arma::mat& X) {
    using arma::trimatl;
    using arma::trimatu;
    using arma::solve;
    return solve(trimatu(L.t()), solve(trimatl(L), X));
}

arma::mat compress_toeplitz(arma::mat& T){
// compress n x n Toeplitz matrix to 2 x n.
    arma::uword n = T.n_rows;
    arma::mat G = arma::zeros<arma::mat>(2, n);
    G.row(0) = T.row(0);
    G.row(1) = T.col(0).t();
    G(1,0) = 0.0;
    return G;
}

arma::mat toep_cholesky_lower(arma::mat& T) {
// lower Cholesky factor of a compressed Toeplitz matrix.
// input: g(1,N), the compressed Toeplitz matrix as col vec
// output L(N,N), the lower Cholesky factor.
    // double scale = T(0,0);
    // T = T / scale;
    arma::mat G = compress_toeplitz(T);
    arma::uword n = G.n_cols;
    arma::mat L = arma::zeros<arma::mat>(n, n);
    arma::mat A = arma::ones<arma::mat>(2, 2);
    double rho = 0.0;
    L.col(0) = G.row(0).t();
    G.row(0).subvec(1,n-1) = G.row(0).subvec(0,n-2);
    G(0,0) = 0;
    for (arma::uword i = 1; i < n; i++)
    {
        rho = -G(1,i)/G(0,i);
        A(0,1) = rho;
        A(1,0) = rho;
        G.submat(0,i,1,n-1) = A * G.submat(0,i,1,n-1) / std::sqrt((1.0-rho)*(1.0+rho));
        L.col(i).subvec(i,n-1) = G.row(0).subvec(i,n-1).t();
        if (i==n-1){break;}
        G.row(0).subvec(i+1,n-1) = G.row(0).subvec(i,n-2);
        G(0,i) = 0.0;
    }

    // T = T * scale;
    return L ; //* std::sqrt(scale)
}

inline arma::mat draw_fstar_(const arma::mat& f, const arma::vec& theta,
                     const arma::vec& theta_star, const arma::mat& beta_prior_sds,
                     const arma::mat& L, const arma::mat& mu_star) {
    // draw fstar for one horizon
    arma::uword n = f.n_rows;
    arma::uword m = f.n_cols;
    arma::uword N = theta_star.n_elem;
    arma::mat result(N, m);
    arma::mat kstar  = K(theta, theta_star, beta_prior_sds.col(0));
    arma::mat kstarT = kstar.t();
    arma::mat tmp = arma::solve(arma::trimatl(L), kstar);
    // arma::mat K_post = K(theta_star, theta_star, 0*beta_prior_sds.col(0));
    // K_post.diag() += 1e-6;
    arma::mat K_post = K(theta_star, theta_star, beta_prior_sds.col(0)) - tmp.t() * tmp;
    K_post.diag() += 1e-6;
    arma::mat L_post = arma::chol(K_post);
    // clock_t begin_time = clock();
    // arma::mat L_post = toep_cholesky_lower(K_post);
    // Rcpp::Rcout << "toeplitz takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
    // // update L_post with rank-1 priors
    // arma::vec update_x;
    // for (arma::uword k = 0; k < beta_prior_sds.n_rows; k++)
    // {
    //     update_x = arma::pow(theta_star,k)*beta_prior_sds(k,0);
    //     L_post = chol_update(L_post, update_x);
    // }
    // Rcpp::Rcout << "updating takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
    
    // for (arma::uword k = 0; k < n; k++)
    // {
    //     update_x = tmp.row(k).t();
    //     bool update = false;
    //     L_post = chol_update(L_post, update_x, update);
    // }
    // Rcpp::Rcout << n << " downdating takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
    // double dist = arma::norm(K_post1 - L_post * L_post.t());
    // Rcpp::Rcout << "dist: " << dist << "\n";

    // begin_time = clock();
    // L_post = arma::chol(K_post1, "lower");
    // Rcpp::Rcout << "chol takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
    // dist = arma::norm(K_post1 - L_post * L_post.t(), 2);
    // Rcpp::Rcout << "dist: " << std::sqrt(dist) << "\n";

    for ( arma::uword j = 0; j < m; ++j ) {
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