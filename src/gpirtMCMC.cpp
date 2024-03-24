#include "gpirt.h"
#include "mvnormal.h"
#include <Rcpp.h>
#include <time.h>

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
                     const arma::mat& theta_prior_means,
                     const arma::mat& theta_prior_sds,
                     const double& theta_os,
                     const double& theta_ls,
                     const std::string& KERNEL,
                     arma::cube thresholds,
                     const int constant_IRF) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword horizon = y.n_slices;
    // arma::uword C = thresholds.n_slices - 1;
    int total_iterations = sample_iterations + burn_iterations;

    // clamp theta
    theta.clamp(-5.0, 5.0);

    // Draw initial values of theta, f, and beta
    arma::mat mean_zeros = arma::zeros<arma::mat>(n, horizon);
    arma::cube S = arma::zeros<arma::cube>(n, n, horizon);
    for (arma::uword h = 0; h < horizon; h++)
    {   
        S.slice(h) = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
        S.slice(h).diag() += 1e-6;
    }
    arma::cube X(n, 3, horizon);
    X.col(0) = arma::ones<arma::mat>(n, horizon);
    X.col(1) = theta;
    X.col(2) = arma::pow(theta,2);
    arma::cube f(n, m, horizon);
    arma::cube cholS(n, n, horizon);
    arma::cube beta(3, m, horizon);

    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean

    arma::cube mu(n,m,horizon);
    Rcpp::Rcout << "Setting up gpirtMCMC...\n";

    // Setup each horizon separately for non-constant IRFs
    if(constant_IRF==0){
        // set up mean
        for(arma::uword h = 0; h < horizon; ++h){
            for ( arma::uword j = 0; j < m; ++j ) {
                for ( arma::uword p = 0; p < 3; ++p ) {
                    beta.slice(h).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
                    // beta.slice(h).col(j).row(p) = 0;
                }
                mu.slice(h) = X.slice(h) * beta.slice(h);
            }
        }
        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword j = 0; j < m; ++j ) {
                // cholS.slice(h) = arma::chol(S.slice(h)+ \
                //             X.slice(h)*arma::diagmat(square(beta_prior_sds.col(j)))* \
                //             X.slice(h).t(), "lower");
                cholS.slice(h) = arma::chol(S.slice(h), "lower");
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
        arma::mat X_constant(n*horizon, 3);
        X_constant.col(0) = arma::ones<arma::vec>(n*horizon);
        X_constant.col(1) = theta_constant;
        X_constant.col(2) = arma::pow(theta_constant,2);
        // Set up S
        arma::mat S_constant = arma::zeros<arma::mat>(n*horizon, n*horizon);
        S_constant = K(theta_constant, theta_constant, beta_prior_sds.col(0));
        S_constant.diag() += 1e-6;

        // Set up cholS/f
        arma::mat f_constant(n*horizon, m);
        arma::mat cholS_constant(n*horizon, n*horizon);
        // cholS_constant = arma::chol(S_constant + \
        //                 X_constant*arma::diagmat(square(beta_prior_sds.col(0)))* \
        //                 X_constant.t(), "lower");
        cholS_constant = arma::chol(S_constant, "lower");
        // set up mean
        arma::mat mu_constant(n*horizon, m);
        for ( arma::uword j = 0; j < m; ++j ) {
            for ( arma::uword p = 0; p < 3; ++p ) {
                beta.slice(0).col(j).row(p) = R::rnorm(beta_prior_means(p, j), beta_prior_sds(p, j));
                // beta.slice(0).col(j).row(p) = 0;
            }
        }
        mu_constant = X_constant * beta.slice(0);
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
                mu.slice(h).col(j) = mu_constant.col(j).subvec(h*n, (h+1)*n-1);
                cholS.slice(h) = cholS_constant.submat(h*n,h*n,(h+1)*n-1,(h+1)*n-1);
            }
        }
    }
    
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 3);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    Xstar.col(2) = arma::pow(theta_star,2);
    arma::cube mu_star(N, m, horizon);
    for (arma::uword h = 0; h < horizon; h++){
        mu_star.slice(h) = Xstar * beta.slice(h);
    }
    
    arma::cube f_star = draw_fstar(f, theta, theta_star, beta_prior_sds,cholS, mu_star, constant_IRF);
    Rcpp::Rcout << "start running gpirtMCMC...\n";

    // Setup results storage
    arma::cube              theta_draws(int(sample_iterations/THIN), n, y.n_slices);
    arma::field<arma::cube> beta_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> f_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> fstar_draws(int(sample_iterations/THIN));
    arma::field<arma::cube> threshold_draws(int(sample_iterations/THIN));
    arma::vec               ll_draws(int(sample_iterations/THIN));

    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;

    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {
        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();

        // set seed
        set_seed(iter);

        // Draw new parameter values
        // clock_t begin_time = clock();
        f      = draw_f(f, theta, y, cholS, beta_prior_sds, mu, thresholds, constant_IRF);
        // Rcpp::Rcout << "drawing f takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
        // begin_time = clock();
        f_star = draw_fstar(f, theta, theta_star,beta_prior_sds, cholS, mu_star, constant_IRF);
        // Rcpp::Rcout << "drawing fstar takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
        // begin_time = clock();
        theta  = draw_theta(theta_star, y, theta, theta_prior_sds, f_star, \
                              mu_star, thresholds, theta_os, theta_ls, KERNEL);
        // Rcpp::Rcout << "drawing theta takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
        // update X from theta
        X.col(1) = theta;
        X.col(2) = arma::pow(theta, 2);

        // Update f for new theta
        arma::mat idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(round(idx(k, h)));
            }
        }
        // draw beta
        // begin_time = clock();
        beta = draw_beta(beta, X, y, f, beta_prior_means, beta_prior_sds, thresholds);
        // Rcpp::Rcout << "drawing beta takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
        
        // update up S, mu, cholS from theta/beta
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta.slice(h);
            mu_star.slice(h) = Xstar * beta.slice(h);
        }
        for (arma::uword h = 0; h < horizon; h++)
        {
            S.slice(h) = K(theta.col(h), theta.col(h), beta_prior_sds.col(0));
            S.slice(h).diag() += 1e-6;
        }

        for (arma::uword h = 0; h < horizon; h++){
            // cholS.slice(h) = arma::chol(S.slice(h)+\
            //         X.slice(h)*arma::diagmat(square(beta_prior_sds.col(1)))*\
            //         X.slice(h).t(), "lower");
            cholS.slice(h) = arma::chol(S.slice(h), "lower");
        }

        // draw thresholds
        // begin_time = clock();
        thresholds = draw_threshold(thresholds, y, f, mu, constant_IRF);
        // Rcpp::Rcout << "drawing threshold takes " << float( clock () - begin_time ) /  CLOCKS_PER_SEC * 1000 << " ms...\n";
        
        // compute current log likelihood
        double ll = 0;
        for (arma::uword h = 0; h < horizon; h++){
            for (arma::uword j = 0; j < m; j++)
            {
                ll += ll_bar(f.slice(h).col(j), y.slice(h).col(j),
                                      mu.slice(h).col(j), thresholds.slice(h).row(j).t());
            }
        }

        if (iter>=burn_iterations && iter%THIN == 0){
            // Store draws
            int store_idx                  = int((iter-burn_iterations)/THIN);
            theta_draws.row(store_idx)     = theta;
            f_draws[store_idx]             = f;
            beta_draws[store_idx]          = beta;
            threshold_draws[store_idx]     = thresholds;
            fstar_draws[store_idx]         = f_star;
            ll_draws[store_idx]            = ll;
        }
    }
    Rprintf("\r100.000 %% complete\n");

    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           Rcpp::Named("f", f_draws),
                                           Rcpp::Named("beta", beta_draws),
                                           Rcpp::Named("fstar", fstar_draws),
                                           Rcpp::Named("threshold", threshold_draws),
                                           Rcpp::Named("ll", ll_draws));
    return result;
}
