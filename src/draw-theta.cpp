#include "gpirt.h"

arma::mat draw_theta(const arma::vec& theta_star,
                     const arma::cube& y, const arma::vec& theta_prior,
                     const arma::cube& fstar, const arma::mat& mu_star,
                     const arma::vec& thresholds,
                     const double& os,
                     const double& ls) {

    // Bookkeeping variables
    arma::uword n = y.n_rows;   // # of respondents 
    arma::uword m = y.n_cols;   // # of item response functions per session
    arma::uword horizon = y.n_slices;  // # of sessions
    arma::uword N = theta_star.n_elem; // # of grids

    // Setup results objects
    arma::mat result(n, horizon);
    arma::vec responses(m);
    arma::vec P(N, arma::fill::zeros);

    // Iterate each respondents
    for ( arma::uword i = 0; i < n; ++i ) {
        // Iterate each session
        for ( arma::uword h = 0; h < horizon; ++h ){
            // For each respondent, extract their responses
            responses = y.slice(h).row(i).t();
            // Setup unnormalized posterior vector
            arma::vec theta_post(N, arma::fill::zeros);
            if(h>0 && os>0){
                // GP: dynamic theta across horizon
                // Compute log likelihood of unnormalized GP posterior over grid
                arma::vec theta_prev = result.row(i).subvec(0,h-1).t();
                arma::vec t_prev     = arma::linspace<arma::vec>(0, h-1, h);
                arma::mat K_prev     = K_time(arma::vec(1, arma::fill::value(h)),t_prev, os, ls);
                arma::mat V          = K_time(t_prev, t_prev, os, ls);
                V.diag()             += 1e-2;
                arma::mat L          = arma::chol(V, "lower");
                arma::mat tmp        = arma::solve(arma::trimatl(L), K_prev.t());
                double v             = os * os - arma::dot(tmp.t(), tmp);
                tmp                  = double_solve(L, theta_prev);
                double product       = arma::dot(K_prev.row(0).t(), tmp);
                arma::vec diff       = theta_star - product;
                theta_post           = (-0.5) * diff % diff / v;
            }
            else{
                // RDM: independent theta
                // Use log likelihood of GP prior over grid
                theta_post = theta_prior;
            }
                
            if(h>0 && ls<0){
                // CST: constant theta across horizon
                // no need to sample for later horizons
                result(i, h) = result(i, 0);
            }
            else if(h==0 && ls<0){
                // CST: sample first theta based on all horizon data
                // sum log likelihoods over all time periods
                P = theta_post;
                for ( arma::uword h = 0; h < horizon; ++h ){
                    for ( arma::uword k = 0; k < N; ++k ) {
                        P[k] += ll_bar(fstar.slice(h).row(k).t(), 
                            y.slice(h).row(i).t(), mu_star.row(k).t(), thresholds);
                    }
                }

                // Add constant value to all entried in the unnormalized ll vector
                arma::vec tmp(2);
                tmp[0] = P.min();
                tmp[1] = -400;
                P = (P - tmp.max()/2);

                // Inverse probability sampling
                P = arma::exp(P);
                P = arma::cumsum(P);
                P = (P - P.min()) / (P.max() - P.min());
                double u = R::runif(0.0, 1.0);
                result(i, h) = theta_star[arma::sum(P<=u)];
            }
            else{
                for ( arma::uword k = 0; k < N; ++k ) {
                    // Then for each value in theta_star,
                    // get log likelihood + log posterior
                    P[k] = theta_post[k] + ll_bar(fstar.slice(h).row(k).t(), 
                                responses, mu_star.row(k).t(), thresholds);
                }

                // Add constant value to all entried in the unnormalized ll vector
                arma::vec tmp(2);
                tmp[0] = P.min();
                tmp[1] = -400;
                P = (P - tmp.max()/2);

                // Exponeniate, cumsum, then scale to [0, 1] for the "CDF"
                P = arma::exp(P);
                P = arma::cumsum(P);
                P = (P - P.min()) / (P.max() - P.min());

                // Then (sort of) inverse sample
                double u = R::runif(0.0, 1.0);
                result(i, h) = theta_star[arma::sum(P<=u)];
            }
        }
    }
    return result;
}
