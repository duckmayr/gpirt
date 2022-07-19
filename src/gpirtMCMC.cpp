#include "gpirt.h"
#include "mvnormal.h"
#include <Rcpp.h>

using namespace Rcpp;

// utility functions to set seed
void set_seed(int seed) {
    Environment base_env("package:base");
    Function set_seed_r = base_env["set.seed"];
    set_seed_r(seed);
}

NumericVector get_seed_state(){
    Rcpp::Environment global_env(".GlobalEnv");
    return global_env[".Random.seed"];
}

void set_seed_state(NumericVector seed_state){
    Rcpp::Environment global_env(".GlobalEnv");
    global_env[".Random.seed"] = seed_state;
}

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::cube& y, arma::mat theta,
                     const int sample_iterations, const int burn_iterations, 
                     const int THIN,
                     const arma::mat& beta_prior_means,
                     const arma::mat& beta_prior_sds,
                     const double& theta_os,
                     const double& theta_ls,
                     arma::vec thresholds,
                     const int constant_IRF) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    arma::uword C = thresholds.n_elem - 1;
    int total_iterations = sample_iterations + burn_iterations;

    // Draw initial values of theta, f, and beta
    arma::mat mean_zeros = arma::zeros<arma::mat>(n, horizon);
    arma::cube S = arma::zeros<arma::cube>(n, n, horizon);
    for (arma::uword h = 0; h < horizon; h++)
    {
        S.slice(h) = K(theta.col(h), theta.col(h));
        S.slice(h).diag() += 1e-6;
    }
    arma::cube X(n, 2, horizon);
    X.col(0) = arma::ones<arma::mat>(n, horizon);
    X.col(1) = theta;
    arma::cube f(n, m, horizon);
    arma::cube cholS(n, n, horizon);

    // Setup each horizon separately for non-constant IRFs
    if(constant_IRF==0){
        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword j = 0; j < m; ++j ) {
                cholS.slice(h) = arma::chol(S.slice(h)+ \
                            X.slice(h)*arma::diagmat(square(beta_prior_sds.col(j)))* \
                            X.slice(h).t(), "lower");
                f.slice(h).col(j) = rmvnorm(cholS.slice(h));
            }
        }
    } 
    else{
        // Setup IRF object jointly using thetas across all horizons
        arma::vec theta_constant(n*horizon);
        for (arma::uword h = 0; h < horizon; h++){
            theta_constant.subvec(h*n, (h+1)*n-1) = theta.col(h);
        }

        // Set up X
        arma::mat X_constant(n*horizon, 2);
        X_constant.col(0) = arma::ones<arma::vec>(n*horizon);
        X_constant.col(1) = theta_constant;
        // Set up S
        arma::mat S_constant = arma::zeros<arma::mat>(n*horizon, n*horizon);
        S_constant = K(theta_constant, theta_constant);
        S_constant.diag() += 1e-6;

        // Set up cholS/f
        arma::mat f_constant(n*horizon, m);
        arma::mat cholS_constant(n*horizon, n*horizon);
        cholS_constant = arma::chol(S_constant + \
                        X_constant*arma::diagmat(square(beta_prior_sds.col(0)))* \
                        X_constant.t(), "lower");

        for ( arma::uword j = 0; j < m; ++j ) {
            // f_constant.col(j) = rmvnorm(cholS_constant);
            // initialize sessions with constant theta/f/fstar
            f_constant.col(j).subvec(0, n-1) = rmvnorm(cholS.slice(0));
            for (arma::uword h = 0; h < horizon; h++){
                f_constant.col(j).subvec(h*n, (h+1)*n-1) = f_constant.col(j).subvec(0, n-1);
            }
        }

        // transfer f_constant into f
        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword j = 0; j < m; ++j ) {
                f.slice(h).col(j) = f_constant.col(j).subvec(h*n, (h+1)*n-1);
                cholS.slice(h) = cholS_constant.submat(h*n,h*n,(h+1)*n-1,(h+1)*n-1);
            }
        }
    }

    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean
    
    arma::cube mu(n,m,horizon);
    for (arma::uword h = 0; h < horizon; h++){
        mu.slice(h) = X.slice(h) * beta_prior_means;
    }
    
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 2);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    arma::mat mu_star = Xstar * beta_prior_means;
    arma::cube f_star = draw_fstar(f, theta, theta_star, cholS, mu_star, constant_IRF);
    
    // The prior probabilities for theta_star doesn't change between iterations
    arma::vec theta_prior(N);
    for ( arma::uword i = 0; i < N; ++i ) {
        theta_prior[i] = R::dnorm(theta_star[i], 0.0, 1.0, 1);
    }

    // Setup results storage
    arma::cube              theta_draws(1+int(sample_iterations/THIN), n, y.n_slices);
    arma::field<arma::cube> f_draws(1+int(sample_iterations/THIN));
    arma::field<arma::cube> fstar_draws(1+int(sample_iterations/THIN));
    arma::mat               threshold_draws(1+int(sample_iterations/THIN), C + 1);

    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;

    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
       
        // set up S, mu, cholS from theta
        for (arma::uword h = 0; h < horizon; h++)
        {
            S.slice(h) = K(theta.col(h), theta.col(h));
            S.slice(h).diag() += 1e-6;
        }
        X.col(1) = theta;
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta_prior_means;
        }
        for (arma::uword h = 0; h < horizon; h++){
            cholS.slice(h) = arma::chol(S.slice(h)+\
                    X.slice(h)*arma::diagmat(square(beta_prior_sds.col(1)))*\
                    X.slice(h).t(), "lower");
        }
        // set seed
        set_seed(iter);

        // Draw new parameter values
        f      = draw_f(f, theta, y, cholS, mu, thresholds, constant_IRF);
        f_star = draw_fstar(f, theta, theta_star, cholS, mu_star, constant_IRF);
        theta  = draw_theta(theta_star, y, theta_prior, f_star, \
                              mu_star, thresholds, theta_os, theta_ls);

        // Update f for new theta
        arma::mat idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(int(idx(k, h)));
            }
        }
        thresholds = draw_threshold(thresholds, y, f, mu);
        if (iter>=(burn_iterations-1) && iter%THIN == 0){
            // Store draws
            int store_idx                  = int((iter+1-burn_iterations)/THIN);
            theta_draws.row(store_idx)     = theta;
            f_draws[store_idx]             = f;
            threshold_draws.row(store_idx) = thresholds.t();
            fstar_draws[store_idx]         = f_star;
        }
    }
    Rprintf("\r100.000 %% complete\n");

    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("f", f_draws),
                                           Rcpp::Named("fstar", fstar_draws),
                                           Rcpp::Named("threshold", threshold_draws));
    return result;
}
