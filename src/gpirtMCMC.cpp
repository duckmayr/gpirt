#include "gpirt.h"
#include "mvnormal.h"
#include <Rcpp.h>

using namespace Rcpp;

// set seed
void set_seed(double seed) {
    Environment base_env("package:base");
    Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
}

NumericVector get_seed_state(){
    Rcpp::Environment global_env(".GlobalEnv");
    return global_env[".Random.seed"];
}

void set_seed_state(NumericVector seed_state){
    Rcpp::Environment global_env(".GlobalEnv");
    global_env[".Random.seed"] = seed_state;
    Environment::Binding v = global_env[".Random.seed"];
    NumericVector state = v;
    Rcpp::Rcout << state << "\n";
}

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::cube& y, arma::mat theta,
                     const int sample_iterations, const int burn_iterations, 
                     const int THIN,
                     const arma::mat& beta_prior_means,
                     const arma::mat& beta_prior_sds,
                     arma::vec thresholds, const int SEED) {
    // set_seed(SEED);
    arma::arma_rng::set_seed(SEED);
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
    for (arma::uword h = 0; h < horizon; h++){
        for ( arma::uword j = 0; j < m; ++j ) {
            cholS.slice(h) = arma::chol(S.slice(h)+ \
                        X.slice(h)*arma::diagmat(square(beta_prior_sds.col(j)))* \
                        X.slice(h).t(), "lower");
            f.slice(h).col(j) = rmvnorm(cholS.slice(h));
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
    arma::cube f_star = draw_fstar(f, theta, theta_star, cholS, mu_star);
    
    // The prior probabilities for theta_star doesn't change between iterations
    arma::vec theta_prior(N);
    for ( arma::uword i = 0; i < N; ++i ) {
        theta_prior[i] = R::dnorm(theta_star[i], 0.0, 1.0, 1);
    }
    // Setup results storage
    arma::cube theta_draws(1+int(sample_iterations/THIN), n, horizon);
    arma::field<arma::cube> f_draws(1+int(sample_iterations/THIN));
    arma::mat threshold_draws(1+int(sample_iterations/THIN), C + 1);
    arma::field<arma::cube> IRFs(int(sample_iterations/THIN));
    arma::mat seed_states(int(sample_iterations/THIN), get_seed_state().length());
    // Information for progress bar:
    double progress_increment = (1.0 / total_iterations) * 100.0;
    double progress = 0.0;
    // Start burn-in loop
    for ( int iter = 0; iter < burn_iterations; ++iter ) {
        // Update progress and check for user interrupt (normally you'd do this
        // and the interrupt check less often, but -- at least for now -- each
        // iteration takes long enough to warrant doing it each time)
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
        // Draw new parameter values
        for (arma::uword h = 0; h < horizon; h++){
            cholS.slice(h) = arma::chol(S.slice(h)+\
                    X.slice(h)*arma::diagmat(square(beta_prior_sds.col(1)))*\
                    X.slice(h).t(), "lower");
        }
        f = draw_f(f, y, cholS, mu, thresholds);
        f_star = draw_fstar(f, theta, theta_star, cholS, mu_star);
        theta = draw_theta(theta_star, y, theta_prior, f_star, mu_star, thresholds);
        X.col(1) = theta;
        // Update f for new theta
        arma::mat idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(int(idx(k, h)));
            }
        }
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta_prior_means;
        }
        thresholds = draw_threshold(thresholds, y, f, mu);
        for (arma::uword h = 0; h < horizon; h++)
        {
            S.slice(h) = K(theta.col(h), theta.col(h));
            S.slice(h).diag() += 1e-6;
        }
    }

    // store initial values
    theta_draws.row(0) = theta;
    f_draws[0] = f;
    threshold_draws.row(0) = thresholds.t();

    // Start sampling loop
    for ( int iter = 0; iter < sample_iterations; ++iter ) {
        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
        // Draw new parameter values
        for (arma::uword h = 0; h < horizon; h++)
        {
            S.slice(h) = K(theta.col(h), theta.col(h));
            S.slice(h).diag() += 1e-6;
        }
        for (arma::uword h = 0; h < horizon; h++){
            cholS.slice(h) = arma::chol(S.slice(h)+\
                    X.slice(h)*arma::diagmat(square(beta_prior_sds.col(1)))*\
                    X.slice(h).t(), "lower");
        }
        // store seed states for recovering fstar
        seed_states.row(int(iter/THIN)) = arma::vec(get_seed_state()).t();
        f = draw_f(f, y, cholS, mu, thresholds);
        f_star = draw_fstar(f, theta, theta_star, cholS, mu_star);
        theta = draw_theta(theta_star, y, theta_prior, f_star, mu_star, thresholds);
        X.col(1) = theta;
        // Update f for new theta
        arma::mat idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            for (arma::uword h = 0; h < horizon; ++h){
                f.slice(h).row(k) = f_star.slice(h).row(int(idx(k, h)));
            }
        }
        for (arma::uword h = 0; h < horizon; h++){
            mu.slice(h) = X.slice(h) * beta_prior_means;
        }
        thresholds = draw_threshold(thresholds, y, f, mu);
        if (iter%THIN == 0){
            // Store draws
            theta_draws.row(1+int(iter/THIN)) = theta;
            f_draws[1+int(iter/THIN)] = f;
            threshold_draws.row(1+int(iter/THIN)) = thresholds.t();
            IRFs[int(iter/THIN)] = f_star;
        }
    }
    Rprintf("\r100.000 %% complete\n");
    // IRFs *= (1.0 / (double)sample_iterations);
    // for ( arma::uword j = 0; j < m; ++j ) {
    //     for ( arma::uword i = 0; i < N; ++i ) {
    //         IRFs(i, j) = R::plogis(IRFs(i, j), 0.0, 1.0, 1, 0);
    //     }
    // }
    Rcpp::List result = Rcpp::List::create(Rcpp::Named("theta", theta_draws),
                                           // Rcpp::Named("beta", beta_draws),
                                           Rcpp::Named("f", f_draws),
                                           Rcpp::Named("threshold", threshold_draws),
                                           Rcpp::Named("IRFs", IRFs),
                                           Rcpp::Named("states", seed_states));
    return result;
}
