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
Rcpp::List gpirtMCMC(const arma::cube& y, arma::mat theta, const int sample_iterations, const int burn_iterations, const int THIN, const arma::mat& beta_prior_means, const arma::mat& beta_prior_sds, const arma::mat& beta_step_sizes, const double& theta_os, const double& theta_ls, arma::cube thresholds, const int constant_IRF);
RcppExport SEXP _gpirt_gpirtMCMC(SEXP ySEXP, SEXP thetaSEXP, SEXP sample_iterationsSEXP, SEXP burn_iterationsSEXP, SEXP THINSEXP, SEXP beta_prior_meansSEXP, SEXP beta_prior_sdsSEXP, SEXP beta_step_sizesSEXP, SEXP theta_osSEXP, SEXP theta_lsSEXP, SEXP thresholdsSEXP, SEXP constant_IRFSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::cube& >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const int >::type sample_iterations(sample_iterationsSEXP);
    Rcpp::traits::input_parameter< const int >::type burn_iterations(burn_iterationsSEXP);
    Rcpp::traits::input_parameter< const int >::type THIN(THINSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_means(beta_prior_meansSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_sds(beta_prior_sdsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_step_sizes(beta_step_sizesSEXP);
    Rcpp::traits::input_parameter< const double& >::type theta_os(theta_osSEXP);
    Rcpp::traits::input_parameter< const double& >::type theta_ls(theta_lsSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< const int >::type constant_IRF(constant_IRFSEXP);
    rcpp_result_gen = Rcpp::wrap(gpirtMCMC(y, theta, sample_iterations, burn_iterations, THIN, beta_prior_means, beta_prior_sds, beta_step_sizes, theta_os, theta_ls, thresholds, constant_IRF));
    return rcpp_result_gen;
END_RCPP
}
// recover_fstar
Rcpp::List recover_fstar(int seed, arma::cube f, const arma::cube& y, const arma::mat& theta, const arma::cube& beta, const arma::cube& thresholds, const arma::mat& beta_prior_means, const arma::mat& beta_prior_sds, const int constant_IRF);
RcppExport SEXP _gpirt_recover_fstar(SEXP seedSEXP, SEXP fSEXP, SEXP ySEXP, SEXP thetaSEXP, SEXP betaSEXP, SEXP thresholdsSEXP, SEXP beta_prior_meansSEXP, SEXP beta_prior_sdsSEXP, SEXP constant_IRFSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< arma::cube >::type f(fSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type thresholds(thresholdsSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_means(beta_prior_meansSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type beta_prior_sds(beta_prior_sdsSEXP);
    Rcpp::traits::input_parameter< const int >::type constant_IRF(constant_IRFSEXP);
    rcpp_result_gen = Rcpp::wrap(recover_fstar(seed, f, y, theta, beta, thresholds, beta_prior_means, beta_prior_sds, constant_IRF));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_gpirt_gpirtMCMC", (DL_FUNC) &_gpirt_gpirtMCMC, 12},
    {"_gpirt_recover_fstar", (DL_FUNC) &_gpirt_recover_fstar, 9},
    {NULL, NULL, 0}
};

RcppExport void R_init_gpirt(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
