#include "gpirt.h"

inline arma::vec marginalize_fstar(const arma::vec& y,
                        const arma::vec& f,
                        const arma::vec& theta,
                        const arma::vec& thresholds){
    arma::vec results(2);
    double approximate_mu = 0;
    double approximate_var = 1;

    // laplace approximation for the marginalized likelihood over fstar

    // return a vector of approximate [mu,var] 
    results[0] = approximate_mu;
    results[1] = approximate_var;
    return results;
}

inline double draw_theta_f_(const arma::vec& y,
                     const arma::mat& f, 
                     const arma::vec& theta,
                     const arma::vec& theta_prior_mu,
                     const arma::vec& theta_prior_var,
                     const arma::vec& thresholds){
    arma::uword m = f.n_rows;   // # of item response functions per session
    // arma::uword horizon = f.n_cols; // # of sessions
    double post_mu = theta_prior_mu[0];
    double post_var = theta_prior_var[0];

    for ( arma::uword j = 0; j < m; ++j ){
        // normal approximation for each item likelihood after marginalizing fstar
        arma::vec normal_lik = marginalize_fstar(y.row(j), f.row(j), theta, thresholds);
        double mu = normal_lik[0];
        double s2 = normal_lik[1];
        // product of two normal pdfs is still a normal pdf
        post_mu = (post_var*mu + s2*post_mu) / (post_var + s2);
        post_var = 1 / (1/post_var + 1/s2);
    }

    // sample from approximate normal posterior
    return R::rnorm(post_mu, std::sqrt(post_var));
}

arma::mat draw_theta_f(const arma::cube& y,
                     const arma::cube& f, const arma::mat& theta,
                     const arma::vec& thresholds,
                     const double& os,
                     const double& ls) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;   // # of respondents 
    arma::uword horizon = y.n_slices;  // # of sessions

    // Setup results objects
    arma::mat result(n, horizon);

    // Iterate each respondents
    for ( arma::uword i = 0; i < n; ++i ) {
        // Iterate each session
        for ( arma::uword h = 0; h < horizon; ++h ){
            // compute prior of x_{it}|x_{it-1},...,x_{i1}
            arma::vec theta_prev = result.row(i).subvec(0,h-1).t();
            arma::vec t_prev     = arma::linspace<arma::vec>(0, h-1, h);
            arma::mat K_prev     = K_time(arma::vec(1, arma::fill::value(h)),t_prev, os, ls);
            arma::mat V          = K_time(t_prev, t_prev, os, ls);
            V.diag()             += 1e-2;
            arma::mat L          = arma::chol(V, "lower");
            arma::mat tmp        = arma::solve(arma::trimatl(L), K_prev.t());
            double v             = os * os - arma::dot(tmp.t(), tmp) + 1e-2;
            tmp                  = double_solve(L, theta_prev);
            double product       = arma::dot(K_prev.row(0).t(), tmp);
            arma::vec theta_prior_mu(1, arma::fill::value(product));
            arma::vec theta_prior_var(1, arma::fill::value(v));
            // sample for x_{it}
            result(i, h) = draw_theta_f_(y.slice(h).row(i).t(), f.row(i),
                    theta.row(i), theta_prior_mu, theta_prior_var, thresholds);
        }
    } 
    return result;
}
