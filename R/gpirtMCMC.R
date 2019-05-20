#' Gaussian Process IRT MCMC
#'
#' Provides posterior samples for the GP-IRT.
#'
#' @param responses A matrix of responses
#' @param sample_iterations An integer vector of length one giving the number
#'   of samples to record
#' @param burn_iterations An integer vector of length one giving the number of
#'   burn in (unrecorded) iterations
#' @param yea_code A vector of length one giving the value corresponding to
#'   a "yea" response; default is 1
#' @param nay_code A vector of length one giving the value corresponding to
#'   a "nay" response; default is 0
#' @param sf A numeric vector of length one giving the scale factor for the
#'   covariance function for the Gaussian process prior; default is 1
#' @param ell A numeric vector of length one giving the length scale for the
#'   covariance function for the Gaussian process prior; default is 1
#'
#' @return A list of length two; the first element, "theta", is a matrix of
#'   dimensions sample_iterations x n giving the theta parameter draws, and the
#'   second element, "f", is an array of dimensions n x m x sample_iterations
#'   giving the f(theta) parameter draws.
#' @export
gpirtMCMC <- function(responses, sample_iterations, burn_iterations,
                      yea_code = 1, nay_code = 0, sf = 1, ell = 1) {
    # First we fix the responses so that yeas are 1 and nays are -1.
    # We copy the responses in case the nay_code is 1 or yea_code is -1
    tmp <- responses
    responses[which(tmp == yea_code)] <- 1
    responses[which(tmp == nay_code)] <- -1
    # Now we can call the C++ sampler function
    result <- .gpirtMCMC(responses, sample_iterations, burn_iterations, sf, ell)
    # And return the result
    return(result)
}
