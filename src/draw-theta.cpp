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

    if ( thresholds.n_cols == f.n_cols ) {
        return ll(f.t() + mu.t(), y, thresholds.t());
    }

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
    int iter = 0;

    while ( reject ) {
        iter += 1;
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


inline arma::vec draw_theta_ess(
        const arma::vec& theta,
        const arma::field<arma::vec>& y,
        const arma::mat& L,
        const arma::field<arma::mat>& fstar,
        const arma::field<arma::mat>& mu_star,
        const arma::field<arma::mat>& thresholds) {
    arma::uword horizon = y.n_rows;

    // First we draw "an ellipse" -- a vector drawn from a multivariate
    // normal with mean zero and covariance Sigma.
    arma::vec nu = rmvnorm(L);
    // Then we calculate the log likelihood threshold for acceptance, "log_y"
    double u = R::runif(0.0,1.0);
    double log_y = std::log(u);
    for (arma::uword h = 0; h < horizon; h++)
    {
        log_y += compute_ll(theta(h), y(h, 0), fstar(h, 0), mu_star(h, 0), thresholds(h, 0));
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
    int iter = 0;

    while ( reject ) {
        iter += 1;
        // Get nu_prime given current epsilon
        theta_prime = theta * std::cos(epsilon) + nu * std::sin(epsilon);
        theta_prime.clamp(-5.0, 5.0);
        // If the log likelihood is over our threshold, accept
        double log_y_prime = 0;
        for (arma::uword h = 0; h < horizon; h++)
        {
            log_y_prime += compute_ll(theta_prime(h), y(h, 0), fstar(h, 0), mu_star(h, 0), thresholds(h, 0));
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

arma::field<arma::vec> draw_theta(
        const arma::vec& theta_star,
        const arma::field<arma::mat>& y,
        const arma::field<arma::vec>& theta,
        const arma::mat& theta_indices,
        const arma::mat& respondent_periods,
        const arma::field<arma::mat>& fstar,
        const arma::field<arma::mat>& mu_star,
        const arma::field<arma::mat>& thresholds,
        const double os, const double ls, const std::string& KERNEL) {

    // Bookkeeping variables
    arma::uword horizon = y.n_rows;
    arma::uword N = theta_star.n_elem;

    // Set up results storage
    arma::field<arma::vec> result(theta);

    // Set theta s.d.s to one
    arma::vec theta_prior_sds = arma::ones(2);

    // Draw new theta values for each respondent
    for ( arma::uword i; i < respondent_periods.n_cols; ++i ) {
        arma::uvec active_indices = arma::find(respondent_periods.col(i) >= 0.0);
        arma::uword T = active_indices.n_elem;
        arma::vec ts(T);
        arma::vec old_value(T);
        arma::field<arma::vec> responses(T);
        arma::field<arma::mat> fstar_subset(T);
        arma::field<arma::mat> mustar_subset(T);
        arma::field<arma::mat> thresholds_subset(T);
        for ( arma::uword j = 0; j < T; ++j ) {
            arma::uword t = active_indices[j];
            ts[j] = respondent_periods(t, i);
            old_value[j] = theta(t, 0)[theta_indices(t, i)];
            responses(j, 0) = y(t, 0).row(theta_indices(t, i)).t();
            fstar_subset(j, 0) = fstar(t, 0);
            mustar_subset(j, 0) = mu_star(t, 0);
            thresholds_subset(j, 0) = thresholds(t, 0);
        }
        arma::mat V = K_time(ts, ts, os, ls, theta_prior_sds, KERNEL);
        V.diag() += 1e-6;
        arma::mat L = arma::chol(V, "lower");
        arma::vec new_draw = draw_theta_ess(
            old_value,
            responses,
            L,
            fstar_subset,
            mustar_subset,
            thresholds_subset
        );
        for ( arma::uword j = 0; j < T; ++j ) {
            arma::uword t = active_indices[j];
            result(t, 0)[theta_indices(t, i)] = new_draw[j];
        }
    }

    return result;
}
