#include "gpirt.h"
#include "mvnormal.h"

inline double compute_ll(const double theta,
                     const arma::vec& y,
                     const arma::mat& fstar,
                     const arma::mat& mu_star,
                     const arma::mat& thresholds){
    // round nu to the nearest index grid
    int theta_index = round((theta+5)/0.01);
    if(theta_index<0){
        theta_index = 0;
    }else if (theta_index>1001)
    {
        theta_index=1001;
    }
    
    arma::rowvec mu = mu_star.row(theta_index);
    arma::rowvec f = fstar.row(theta_index);
    
    // compute log likelihood
    return ll(f.t()+mu.t(), y, thresholds);
}

inline arma::vec draw_theta_ess(const arma::vec& theta,
                     const arma::mat& y,
                     const arma::mat& L,
                     const arma::cube& fstar,
                     const arma::cube& mu_star,
                     const arma::cube& thresholds){
    arma::uword horizon = y.n_cols;

    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec nu = rmvnorm(L);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0,1.0);
    double log_y = std::log(u);
    for (arma::uword h = 0; h < horizon; h++)
    {
        log_y += compute_ll(theta(h), y.col(h), fstar.slice(h),\
                     mu_star.slice(h), thresholds.slice(h));
    }

    // For our while loop condition:
    bool reject = true;
    // Set up the proposal band and draw initial proposal epsilon:
    double epsilon_min = 0.0;
    double epsilon_max = M_2PI;
    double epsilon = R::runif(epsilon_min, epsilon_max);
    // double epsilon = (epsilon_max-epsilon_min)*arma::randu(1) + epsilon_min;
    epsilon_min = epsilon - M_2PI;
    // We'll create the arma::vec object for nu_prime out of the loop
    arma::vec theta_prime(horizon, arma::fill::zeros);

    while ( reject ) {
        // Get nu_prime given current epsilon
        theta_prime = theta * std::cos(epsilon) + nu * std::sin(epsilon);
        theta_prime.clamp(-5.0, 5.0);
        // If the log likelihood is over our threshold, accept
        double log_y_prime = 0;
        for (arma::uword h = 0; h < horizon; h++)
        {
            log_y_prime += compute_ll(theta_prime(h), y.col(h),\
                         fstar.slice(h), mu_star.slice(h), thresholds.slice(h));
        }

        if ( log_y_prime > log_y ) {
            reject = false;
        }
        // otw, adjust our proposal band & draw a new epsilon, then repeat
        else {
            if ( epsilon < 0.0 ) {
                epsilon_min = epsilon;
            }
            else {
                epsilon_max = epsilon;
            }
            epsilon = R::runif(epsilon_min, epsilon_max);
        }
    }
    return theta_prime;
}

arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::mat& theta,
                     const arma::mat& theta_prior_sds,
                     const arma::cube& fstar, const arma::cube& mu_star,
                     const arma::cube& thresholds,
                     const double& os,
                     const double& ls, const std::string& KERNEL) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;   // # of respondents 
    arma::uword m = y.n_cols;   // # of item response functions per session
    arma::uword horizon = y.n_slices;  // # of sessions
    arma::uword N = theta_star.n_elem; // # of grids

    // Setup results objects
    arma::mat result(n, horizon);

    // compute cholesky decomposition
    arma::vec ts       = arma::linspace<arma::vec>(0, horizon-1, horizon);
    arma::mat V; 
    if(ls>=3*horizon){
        // CST: constant theta across horizon
        arma::mat f_constant(m*horizon, N);
        arma::mat y_constant(m*horizon, n);
        arma::mat mu_constant(m*horizon, N);
        for (arma::uword h = 0; h < horizon; h++){
            for ( arma::uword k = 0; k < N; ++k ) {
                f_constant.col(k).subvec(h*m, (h+1)*m-1) = fstar.slice(h).row(k).t();
                mu_constant.col(k).subvec(h*m, (h+1)*m-1) = mu_star.slice(h).row(k).t();
            }
            for (arma::uword i = 0; i < n; ++i )
            {
                y_constant.col(i).subvec(h*m, (h+1)*m-1) = y.slice(h).row(i).t();
            }
            
        }
        V.ones(1, 1);
        for ( arma::uword i = 0; i < n; ++i ){
            arma::mat y_(m*horizon,1);
            arma::cube fstar_(N, m*horizon,1);
            arma::cube mu_star_(N, m*horizon,1);
            y_.col(0) = y_constant.col(i);
            fstar_.slice(0) = f_constant.t();
            mu_star_.slice(0) = mu_constant.t();
            arma::vec raw_theta_ess = draw_theta_ess(arma::vec(1, arma::fill::value(theta(i,0))), y_, \
                        arma::chol(V+std::pow(theta_prior_sds(0,i),2), "lower"), fstar_, mu_star_, thresholds);
            for ( arma::uword h = 0; h < horizon; ++h ){
                result(i, h) = theta_star[round((raw_theta_ess(0)+5)/0.01)];
            }
        }
    }else if(ls<=0.1){
        // RDM: independent theta
        V.ones(1, 1);
        for ( arma::uword i = 0; i < n; ++i ){
            // round up theta to nearest fine grid
            for ( arma::uword h = 0; h < horizon; ++h ){
                arma::mat y_(m, 1);
                y_.col(0) = y.slice(h).row(i).t();
                arma::cube fstar_(N, m, 1);
                arma::cube mu_star_(N, m, 1);
                fstar_.slice(0) = fstar.slice(h);
                mu_star_.slice(0) = mu_star.slice(h);
                arma::vec raw_theta_ess = draw_theta_ess(arma::vec(1, arma::fill::value(theta(i,h))),\
                       y_, arma::chol(V+std::pow(theta_prior_sds(0,i),2), "lower"), fstar_, mu_star_, thresholds);
                result(i, h) = theta_star[round((raw_theta_ess(0)+5)/0.01)];
            }
        }
    }else{
        // Iterate each respondents
        for ( arma::uword i = 0; i < n; ++i ){
            V              = K_time(ts, ts, os, ls, theta_prior_sds.col(i), KERNEL);
            V.diag()       += 1e-6;
            arma::mat L    = arma::chol(V, "lower");
            // draw dynamic theta using ess
            arma::vec raw_theta_ess = draw_theta_ess(theta.row(i).t(), y.row(i), L, fstar, mu_star, thresholds);
            // round up theta to nearest fine grid
            for ( arma::uword h = 0; h < horizon; ++h ){
                result(i, h) = theta_star[round((raw_theta_ess(h)+5)/0.01)];
            }
        }
    }
    
    return result;
}

// arma::mat draw_theta(const arma::vec& theta_star,
//                      const arma::cube& y, const arma::vec& theta_prior,
//                      const arma::cube& fstar, const arma::cube& mu_star,
//                      const arma::vec& thresholds,
//                      const double& os,
//                      const double& ls) {

//     // Bookkeeping variables
//     arma::uword n = y.n_rows;   // # of respondents 
//     // arma::uword m = y.n_cols;   // # of item response functions per session
//     arma::uword horizon = y.n_slices;  // # of sessions
//     arma::uword N = theta_star.n_elem; // # of grids

//     // Setup results objects
//     arma::mat result(n, horizon);
//     arma::vec P(N, arma::fill::zeros);

//     // Iterate each respondents
//     for ( arma::uword i = 0; i < n; ++i ) {
//         // Iterate each session
//         for ( arma::uword h = 0; h < horizon; ++h ){
//             // Setup unnormalized posterior vector
//             arma::vec theta_post(N, arma::fill::zeros);
//             if(h>0 && os>0){
//                 // GP: dynamic theta across horizon
//                 // Compute log likelihood of unnormalized GP posterior over grid
//                 arma::vec theta_prev = result.row(i).subvec(0,h-1).t();
//                 arma::vec t_prev     = arma::linspace<arma::vec>(0, h-1, h);
//                 arma::mat K_prev     = K_time(arma::vec(1, arma::fill::value(h)),t_prev, os, ls);
//                 arma::mat V          = K_time(t_prev, t_prev, os, ls);
//                 V.diag()             += 1e-2;
//                 arma::mat L          = arma::chol(V, "lower");
//                 arma::mat tmp        = arma::solve(arma::trimatl(L), K_prev.t());
//                 double v             = os * os - arma::dot(tmp.t(), tmp) + 1e-2;
//                 tmp                  = double_solve(L, theta_prev);
//                 double product       = arma::dot(K_prev.row(0).t(), tmp);
//                 arma::vec diff       = theta_star - product;
//                 theta_post           = (-0.5) * diff % diff / v;
//                 // arma::vec theta_other = result.shed_col(h).row(i).t();
//                 // arma::vec t_other     = arma::linspace<arma::vec>(0, horizon-1, horizon);
//                 // t_other              = t_other.shed_row(h);
//                 // arma::mat K_other    = K_time(arma::vec(1, arma::fill::value(h)),t_other, os, ls);
//                 // arma::mat V          = K_time(t_other, t_other, os, ls);
//                 // V.diag()             += 1e-4;
//                 // arma::mat L          = arma::chol(V, "lower");
//                 // arma::mat tmp        = arma::solve(arma::trimatl(L), K_other.t());
//                 // double v             = os * os - arma::dot(tmp.t(), tmp);
//                 // tmp                  = double_solve(L, theta_other);
//                 // double product       = arma::dot(K_other.row(0).t(), tmp);
//                 // arma::vec diff       = theta_star - product;
//                 // theta_post           = (-0.5) * diff % diff / v;

//             }
//             else{
//                 // RDM: independent theta
//                 // Use log likelihood of GP prior over grid
//                 theta_post = theta_prior;
//             }
                
//             if(h>0 && ls<0){
//                 // CST: constant theta across horizon
//                 // no need to sample for later horizons
//                 result(i, h) = result(i, 0);
//             }
//             else if(h==0 && ls<0){
//                 // CST: sample first theta based on all horizon data
//                 // sum log likelihoods over all time periods
//                 P = theta_post;
//                 for ( arma::uword h = 0; h < horizon; ++h ){
//                     for ( arma::uword k = 0; k < N; ++k ) {
//                         P[k] += ll_bar(fstar.slice(h).row(k).t(), 
//                             y.slice(h).row(i).t(), mu_star.slice(h).row(k).t(), thresholds);
//                     }
//                 }

//                 // Add constant value to all entried in the unnormalized ll vector
//                 arma::vec tmp(2);
//                 tmp[0] = P.min();
//                 tmp[1] = -400;
//                 P = (P - tmp.max()/2);

//                 // Inverse probability sampling
//                 P = arma::exp(P);
//                 P = arma::cumsum(P);
//                 P = (P - P.min()) / (P.max() - P.min());
//                 double u = R::runif(0.0, 1.0);
//                 result(i, h) = theta_star[arma::sum(P<=u)];
//             }
//             else{
//                 P = theta_post;
//                 for ( arma::uword k = 0; k < N; ++k ) {
//                     // Then for each value in theta_star,
//                     // get log likelihood + log posterior
//                     P[k] += ll_bar(fstar.slice(h).row(k).t(), 
//                                 y.slice(h).row(i).t(), mu_star.slice(h).row(k).t(), thresholds);
//                 }

//                 // Add constant value to all entried in the unnormalized ll vector
//                 arma::vec tmp(2);
//                 tmp[0] = P.min();
//                 tmp[1] = -400;
//                 P = (P - tmp.max()/2);

//                 // Exponeniate, cumsum, then scale to [0, 1] for the "CDF"
//                 P = arma::exp(P);
//                 P = arma::cumsum(P);
//                 P = (P - P.min()) / (P.max() - P.min());

//                 // Then (sort of) inverse sample
//                 double u = R::runif(0.0, 1.0);
//                 result(i, h) = theta_star[arma::sum(P<=u)];
//             }
//         }
//     }
//     return result;
// }
