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
Rcpp::List gpirtMCMC(
        const arma::field<arma::mat>& y,
        arma::field<arma::vec> theta,
        const arma::mat& theta_indices,
        const arma::mat& respondent_periods,
        const int sample_iterations,
        const int burn_iterations,
        const int THIN,
        const arma::mat& beta_prior_means,
        const arma::mat& beta_prior_sds,
        const arma::mat& theta_prior_means,
        const arma::mat& theta_prior_sds,
        const double& theta_os,
        const double& theta_ls,
        const std::string& KERNEL,
        arma::field<arma::mat> thresholds,
        const int constant_IRF
        ) {

    Rcpp::Rcout << "Setting initial parameter values...\n";

    // Bookkeeping variables
    arma::uword horizon = y.n_rows;
    int total_iterations = sample_iterations + burn_iterations;

    // Clamp theta to [-5, 5]
    theta.for_each( [](arma::vec& v) { v.clamp(-5.0, 5.0); } );

    // Draw initial values of theta, f, and beta
    arma::field<arma::vec> mean_zeroes(horizon);
    for ( arma::uword h = 0; h < horizon; ++h ) {
        mean_zeroes(h, 0) = arma::zeros(y(h, 0).n_rows);
    }
    arma::field<arma::mat> S(horizon);
    for ( arma::uword h = 0; h < horizon; h++ ) {
        S(h, 0) = K(theta(h, 0), theta(h, 0), arma::ones(3)); // beta_prior_sds.col(0));
        S(h, 0).diag() += 1e-6;
    }
    arma::field<arma::mat> X(horizon);
    for ( arma::uword h = 0; h < horizon; h++ ) {
        X(h, 0) = arma::join_horiz(
            arma::ones(y(h, 0).n_rows),
            theta(h, 0),
            arma::pow(theta(h, 0), 2.0)
        );
    }
    arma::field<arma::mat> f(horizon);
    arma::field<arma::mat> cholS(horizon);
    arma::field<arma::mat> beta(horizon);

    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean
    arma::field<arma::mat> mu(horizon);

    // Setup each horizon separately for non-constant IRFs
    if ( constant_IRF == 0 ) {
        // Set up mean
        for ( arma::uword h = 0; h < horizon; ++h ) {
            beta(h, 0) = arma::zeros(3, y(h, 0).n_cols);
            mu(h, 0) = X(h, 0) * beta(h, 0);
            cholS(h, 0) = arma::chol(S(h, 0), "lower");
            arma::uword n = y(h, 0).n_rows;
            arma::uword m = y(h, 0).n_cols;
            arma::mat tmp(n, m);
            for ( arma::uword j = 0; j < m; ++j ) {
                tmp.col(j) = rmvnorm(cholS(h, 0));
            }
            f(h, 0) = tmp;
        }

    } else {
        Rcpp::stop("Constant IRF not yet implemented.");
        /*
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
        */
    }

    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 3);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    Xstar.col(2) = arma::pow(theta_star,2);
    arma::field<arma::mat> mu_star(horizon);
    for ( arma::uword h = 0; h < horizon; ++h ) {
        mu_star(h, 0) = Xstar * beta(h, 0);
    }

    // Draw initial f*
    arma::field<arma::mat> f_star = draw_fstar(
        f, theta, theta_star, beta_prior_sds, cholS, mu_star, constant_IRF
    );

    // Setup results storage
    Rcpp::Rcout << "Setting up result storage...\n";
    arma::uword ndraws = int(sample_iterations/THIN);
    arma::field<arma::mat>  theta_draws(horizon);
    arma::field<arma::cube>  beta_draws(horizon);
    arma::field<arma::cube>     f_draws(horizon);
    arma::field<arma::cube> fstar_draws(horizon);
    arma::field<arma::cube> threshold_draws(horizon);
    arma::uword C = thresholds(0, 0).n_rows;
    for ( arma::uword h = 0; h < horizon; ++h ) {
        arma::uword n = y(h, 0).n_rows;
        arma::uword m = y(h, 0).n_cols;
        theta_draws(h, 0) = arma::zeros(n, ndraws);
        beta_draws(h, 0) = arma::zeros(3, m, ndraws);
        f_draws(h, 0) = arma::zeros(n, m, ndraws);
        fstar_draws(h, 0) = arma::zeros(N, m, ndraws);
        threshold_draws(h, 0) = arma::zeros(C, m, ndraws);
    }
    arma::vec ll_draws(int(sample_iterations/THIN));

    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;
    Rcpp::Rcout << "Sampling...\n";

    // Start sampling loop
    for ( int iter = 0; iter < total_iterations; ++iter ) {

        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();

        // Draw new parameter values
        f      = draw_f(f, theta, y, cholS, beta_prior_sds, mu, thresholds, constant_IRF);
        f_star = draw_fstar(f, theta, theta_star,beta_prior_sds, cholS, mu_star, constant_IRF);
        theta  = draw_theta(
            theta_star, y, theta, theta_indices, respondent_periods,
            f_star, mu_star, thresholds, theta_os, theta_ls, KERNEL
        );

        // Update X from theta
        for ( arma::uword h = 0; h < horizon; h++ ) {
            X(h, 0).col(1) = theta(h, 0);
            X(h, 0).col(2) = arma::pow(theta(h, 0), 2.0);
        }

        // Update S, mu, cholS from theta/beta
        for ( arma::uword h = 0; h < horizon; ++h ){
            mu(h, 0) = X(h, 0) * beta(h, 0);
            mu_star(h, 0) = Xstar * beta(h, 0);
            // S(h, 0) = K(theta(h, 0), theta(h, 0), beta_prior_sds.col(0));
            S(h, 0) = K(theta(h, 0), theta(h, 0), arma::ones(3));
            S(h, 0).diag() += 1e-6;
            cholS(h, 0) = arma::chol(S(h, 0), "lower");
        }

        // Draw thresholds
        thresholds = draw_threshold(thresholds, y, f, mu, constant_IRF);

        // Compute current log likelihood
        double ll = 0;
        for (arma::uword h = 0; h < horizon; h++){
            arma::uword m = f(h, 0).n_cols;
            for ( arma::uword j = 0; j < m; ++j ) {
                ll += ll_bar(
                    f(h, 0).col(j), y(h, 0).col(j), mu(h, 0).col(j),
                    thresholds(h, 0).col(j)
                );
            }
        }

        // Store draws
        if ( iter >= burn_iterations && iter % THIN == 0 ) {
            int store_idx       = int((iter-burn_iterations)/THIN);
            ll_draws[store_idx] = ll;
            for ( arma::uword h = 0; h < horizon; ++h ) {
                theta_draws(h, 0).col(store_idx)   = theta(h, 0);
                f_draws(h, 0).slice(store_idx)     = f(h, 0);
                beta_draws(h, 0).slice(store_idx)  = beta(h, 0);
                fstar_draws(h, 0).slice(store_idx) = f_star(h, 0);
                threshold_draws(h, 0).slice(store_idx) = thresholds(h, 0);
            }
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
