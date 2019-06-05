#' Gaussian Process IRT MCMC
#'
#' Provides posterior samples for the GP-IRT.
#'
#' @param data An object of class \code{\link{response_matrix}},
#'   or an object coercible to class \code{response_matrix}
#' @param sample_iterations An integer vector of length one giving the number
#'   of samples to record
#' @param burn_iterations An integer vector of length one giving the number of
#'   burn in (unrecorded) iterations
#' @param yea_codes A vector giving the values corresponding to a "yea"
#'   response (default is 1); only used if \code{data} must be coerced to a
#'   \code{response_matrix} object
#' @param nay_codes A vector giving the values corresponding to a "nay"
#'   response (default is 0); only used if \code{data} must be coerced to a
#'   \code{response_matrix} object
#' @param missing_codes A vector giving the values corresponding to a missing
#'   response (default is NA); only used if \code{data} must be coerced to
#'   \code{response_matrix} object
#' @param sf A numeric vector of length one giving the scale factor for the
#'   covariance function for the Gaussian process prior; default is 1
#' @param ell A numeric vector of length one giving the length scale for the
#'   covariance function for the Gaussian process prior; default is 1
#'
#' @return A list of length two; the first element, "theta", is a matrix of
#'   dimensions sample_iterations x n giving the theta parameter draws, and the
#'   second element, "f", is an array of dimensions n x m x sample_iterations
#'   giving the f(theta) parameter draws.
#'
#' @examples
#' ilogit <- function(x) 1 / (1 + exp(-x)) # inverse logit function
#' ##### Monotonic IRT example ####
#' ## Simulate data
#' gen_responses <- function(theta, alpha, beta) {
#'     # Stardard 2PL model
#'     n <- length(theta)
#'     m <- length(alpha)
#'     responses <- matrix(0, n, m)
#'     for ( j in 1:m ) {
#'         for ( i in 1:n ) {
#'             p <- ilogit(alpha[j] + beta[j] * theta[i])
#'             responses[i, j] <- sample(0:1, 1, prob = c(1 - p, p))
#'         }
#'     }
#'     return(responses)
#' }
#' set.seed(1234)
#' n <- 30
#' m <- 10
#' theta <- seq(-3, 3, length.out = n) # Respondent ability parameters
#' alpha <- seq(-2, 2, length.out = m) # Item difficulty parameters
#' beta  <- runif(m, 0.5, 3)           # Item discrimination parameters
#' responses <- gen_responses(theta, alpha, beta)
#'
#' ## Check for unanimity and omit any unanimous items
#' table(apply(responses, 2, function(x) length(unique(x))))
#' unanimous_items <- which(apply(responses, 2, function(x) length(unique(x))) < 2)
#' N <- length(unanimous_items)
#' if ( N == 0 ) unanimous_items <- n + 1 else m <- m - N
#' responses <- responses[ , -unanimous_items]
#'
#' ## Generate samples
#' ## (We just use 1 iteration for a short-running toy example here;
#' ##  try 500-1000+ to fully demo the sampler's behavior)
#' samples <- gpirtMCMC(responses, 1, 0)
#' str(samples)
#'
#' @export
gpirtMCMC <- function(data, sample_iterations, burn_iterations,
                      yea_codes = 1, nay_codes = 0, missing_codes = NA,
                      sf = 1, ell = 1) {
    # First we make sure our data are in the proper format:
    data <- as.response_matrix(data, yea_codes, nay_codes, missing_codes)
    # Now we can call the C++ sampler function
    result <- .gpirtMCMC(data, sample_iterations, burn_iterations, sf, ell)
    # And return the result
    return(result)
}
