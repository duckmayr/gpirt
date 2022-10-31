#include <RcppArmadillo.h>

arma::mat K(const arma::vec& x1, const arma::vec& x2,
            const arma::vec& beta_prior_sds) {
    arma::uword n = x1.n_elem;
    arma::uword m = x2.n_elem;
    arma::mat result(n, m);
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double diff = x1[i] - x2[j];
            result(i, j) = std::exp(-0.5 * diff * diff);
            result(i, j) += x1(i) * std::pow(beta_prior_sds(1),2) * x2(j);
            result(i, j) += std::pow(beta_prior_sds(0),2);
            result(i, j) += std::pow(x1(i)*beta_prior_sds(2)*x2(j),2);
        }
    }
    return result;
}


arma::mat K_time(const arma::vec& x1, const arma::vec& x2, 
                 const double& os, const double& ls, 
                 const arma::vec& theta_prior_sds, const std::string& KERNEL){
    // Compute the Matérn covariance of degree 5/2
    arma::uword n = x1.n_elem;
    arma::uword m = x2.n_elem;
    arma::mat result(n, m); 
    for ( arma::uword j = 0; j < m; ++j ) {
        for ( arma::uword i = 0; i < n; ++i ) {
            double diff = std::abs(x1(i) - x2(j));
            if (KERNEL.compare("Matern")==0){
                result(i, j) = os * os * (1 + std::sqrt(5) * diff / ls + 5 *  diff * diff / ls / ls / 3);
                result(i, j) = result(i, j) * std::exp(- std::sqrt(5) * diff / ls);
                result(i, j) += x1(i) * std::pow(theta_prior_sds(1),2) * x2(j);
                result(i, j) += std::pow(theta_prior_sds(0),2);
            }else if (KERNEL.compare("RBF")==0){
                result(i, j) = os * os * std::exp(- diff * diff / ls / ls);
                result(i, j) += x1(i) * std::pow(theta_prior_sds(1),2) * x2(j);
                result(i, j) += std::pow(theta_prior_sds(0),2);
            }
        }
    }
    return result;
}
