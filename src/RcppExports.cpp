// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// gpirtMCMC
Rcpp::List gpirtMCMC(const arma::cube& y, arma::mat theta, const int sample_iterations, const int burn_iterations, const arma::mat& fix_theta_flag, const arma::mat& fix_theta_value, const int THIN, const arma::mat& beta_prior_means, const arma::mat& beta_prior_sds, const double& theta_os, const double& theta_ls, arma::vec thresholds);
RcppExport SEXP _gpirt_gpirtMCMC(SEXP ySEXP, SEXP thetaSEXP, SEXP sample_iterationsSEXP, SEXP burn_iterationsSEXP, SEXP fix_theta_flagSEXP, SEXP fix_theta_valueSEXP, SEXP THINSEXP, SEXP beta_prior_meansSEXP, SEXP beta_prior_sdsSEXP, SEXP theta_osSEXP, SEXP theta_lsSEXP, SEXP thresholdsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const int >::type sample_iterations(sample_iterationsSEXP);
    Rcpp::traits::input_parameter< const int >::type burn_iterations(burn_iterationsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type fix_theta_flag(fix_theta_flagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type fix_theta_value(fix_theta_valueSEXP);
    Rcpp::traits::input_parameter< const int >::type THIN(THINSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_means(beta_prior_meansSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_sds(beta_prior_sdsSEXP);
    Rcpp::traits::input_parameter< const double& >::type theta_os(theta_osSEXP);
    Rcpp::traits::input_parameter< const double& >::type theta_ls(theta_lsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type thresholds(thresholdsSEXP);
    rcpp_result_gen = Rcpp::wrap(gpirtMCMC(y, theta, sample_iterations, burn_iterations, fix_theta_flag, fix_theta_value, THIN, beta_prior_means, beta_prior_sds, theta_os, theta_ls, thresholds));
    return rcpp_result_gen;
END_RCPP
}
// recover_fstar
Rcpp::List recover_fstar(int seed, arma::cube f, const arma::cube& y, const arma::mat& theta, const arma::vec& thresholds, const arma::mat& beta_prior_means, const arma::mat& beta_prior_sds);
RcppExport SEXP _gpirt_recover_fstar(SEXP seedSEXP, SEXP fSEXP, SEXP ySEXP, SEXP thetaSEXP, SEXP thresholdsSEXP, SEXP beta_prior_meansSEXP, SEXP beta_prior_sdsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type f(fSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_means(beta_prior_meansSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_sds(beta_prior_sdsSEXP);
    rcpp_result_gen = Rcpp::wrap(recover_fstar(seed, f, y, theta, thresholds, beta_prior_means, beta_prior_sds));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gpirt_gpirtMCMC", (DL_FUNC) &_gpirt_gpirtMCMC, 12},
    {"_gpirt_recover_fstar", (DL_FUNC) &_gpirt_recover_fstar, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_gpirt(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
