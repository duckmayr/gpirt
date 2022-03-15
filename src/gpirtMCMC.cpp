#include "gpirt.h"
#include "mvnormal.h"

// set seed
// [[Rcpp::export]]
void set_seed(double seed) {
    Rcpp::Environment base_env("package:base");
    Rcpp::Function set_seed_r = base_env["set.seed"];
    set_seed_r(std::floor(std::fabs(seed)));
}

// [[Rcpp::export(.gpirtMCMC)]]
Rcpp::List gpirtMCMC(const arma::mat& y, arma::vec theta,
                     const int sample_iterations, const int burn_iterations, 
                     const int THIN,
                     const arma::mat& beta_prior_means,
                     const arma::mat& beta_prior_sds,
                     arma::vec thresholds, const int SEED) {
    set_seed(SEED);
    arma::uword n = y.n_rows;
    arma::uword m = y.n_cols;
    arma::uword C = thresholds.n_elem - 1;
    int total_iterations = sample_iterations + burn_iterations;
    // Draw initial values of theta, f, and beta
    arma::vec mean_zeros = arma::zeros<arma::vec>(n);
    arma::mat S = K(theta, theta);
    S.diag() += 1e-6;
    arma::mat X(n, 2);
    X.col(0) = arma::ones<arma::vec>(n);
    X.col(1) = theta;
    arma::mat f(n, m);
    arma::mat cholS = arma::chol(S, "lower");
    for ( arma::uword j = 0; j < m; ++j ) {
        cholS = arma::chol(S+X*arma::diagmat(square(beta_prior_sds.col(j)))*X.t(), "lower");
        f.col(j) = rmvnorm(cholS);
    }
   
    // We need to have a matrix with a column of ones and a column of theta
    // for generating the linear mean
    
    arma::mat mu =  X * beta_prior_means;
    // Setup theta_star grid
    arma::vec theta_star = arma::regspace<arma::vec>(-5.0, 0.01, 5.0);
    arma::uword N = theta_star.n_elem;
    arma::mat Xstar(N, 2);
    Xstar.col(0) = arma::ones<arma::vec>(N);
    Xstar.col(1) = theta_star;
    arma::mat mu_star = Xstar * beta_prior_means;
    arma::mat f_star  = draw_fstar(f, theta, theta_star, cholS, mu_star);
    
    // The prior probabilities for theta_star doesn't change between iterations
    arma::vec theta_prior(N);
    for ( arma::uword i = 0; i < N; ++i ) {
        theta_prior[i] = R::dnorm(theta_star[i], 0.0, 1.0, 1);
    }
    // Setup results storage
    arma::mat theta_draws(int(sample_iterations/THIN), n);
    arma::cube f_draws(n, m, int(sample_iterations/THIN));
    arma::mat threshold_draws(int(sample_iterations/THIN), C+1);
    arma::cube IRFs(N, m, int(sample_iterations/THIN), arma::fill::zeros);
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
        cholS = arma::chol(S+X*arma::diagmat(square(beta_prior_sds.col(1)))*X.t(), "lower");
        f = draw_f(f, y, cholS, mu, thresholds);
        f_star = draw_fstar(f, theta, theta_star, cholS, mu_star);
        theta = draw_theta(theta_star, y, theta_prior, f_star, mu_star, thresholds);
        X.col(1) = theta;
        // Update f for new theta
        arma::vec idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            f.row(k) = f_star.row(int(idx[k]));
        }
        thresholds = draw_threshold(thresholds, y, f, mu);
        mu = X * beta_prior_means;
        mu_star = Xstar * beta_prior_means;
        S = K(theta, theta);
        S.diag() += 1e-6;
    }
    // Start sampling loop
    for ( int iter = 0; iter < sample_iterations; ++iter ) {
        // Update progress and check for user interrupt
        Rprintf("\r%6.3f %% complete", progress);
        progress += progress_increment;
        Rcpp::checkUserInterrupt();
        // Draw new parameter values
        cholS = arma::chol(S+X*arma::diagmat(square(beta_prior_sds.col(1)))*X.t(), "lower");
        f = draw_f(f, y, cholS, mu, thresholds);
        f_star = draw_fstar(f, theta, theta_star, cholS, mu_star);
        theta = draw_theta(theta_star, y, theta_prior, f_star, mu_star, thresholds);
        X.col(1) = theta;
        // Update f for new theta
        arma::vec idx = (theta+5)/0.01;
        for (arma::uword k = 0; k < n; ++k){
            f.row(k) = f_star.row(int(idx[k]));
        }
        thresholds = draw_threshold(thresholds, y, f, mu);
        mu = X * beta_prior_means;
        mu_star = Xstar * beta_prior_means;
        S = K(theta, theta);
        S.diag() += 1e-6;
        if (iter%THIN == 0){
            // Store draws
            theta_draws.row(int(iter/THIN)) = theta.t();
            // beta_draws.slice(iter) = beta;
            f_draws.slice(int(iter/THIN)) = f;
            threshold_draws.row(int(iter/THIN)) = thresholds.t();
            // Update IRF estimates
            IRFs.slice(int(iter/THIN)) = f_star;
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
                                           Rcpp::Named("IRFs", IRFs));
    return result;
}
