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
#' @param vote_codes A named list giving the mapping from recorded responses to
#'   {-1, 1, NA}. An element named "yea" gives the responses that should be
#'   coded as 1, an element named "nay" gives the responses that should be coded
#'   as -1, and an element named "missing" gives responses that should be NA;
#'   only used if \code{data} must be coerced to a \code{response_matrix} object.
#'   If NULL, then ordinal regression will be performed.
#' @param beta_prior_means A numeric matrix of with \code{ncol(data)} columns
#'   and two rows giving the prior means for the items' linear means' intercept
#'   and slope; by default, a matrix of zeros
#' @param beta_prior_sds A numeric matrix of with \code{ncol(data)} columns
#'   and two rows giving the prior standard deviations for the items' linear
#'   means' intercept and slope; by default, a matrix of threes
#' @param beta_proposal_sds A numeric matrix of with \code{ncol(data)} columns
#'   and two rows giving the standard deviations for proposals for the items'
#'   linear means' intercept and slope; by default a matrix filled with 0.1
#' @param theta_init A vector of length \code{nrow(data)} giving initial values
#'   for the respondent ideology parameters; if NULL (the default), the initial
#'   values are drawn from the parameters' prior distributions.
#' @param thresholds A vector of length \code{C+1} where \code{C} is the number
#'   of all categories; if NULL (the default), the values are chosen from -Inf to Inf
#'    of length \code{C+1} with each y having equal probablity under prior mean.
#'
#' @return A list with elements
#'   \describe{
#'       \item{theta}{The theta parameter draws, stored in a matrix with
#'                    \code{sample_iterations} + 1 rows (initial values are
#'                    included) and n columns.}
#'       \item{beta}{The beta parameter draws, stored in an array with 2 rows,
#'                   \code{m} columns, and \code{sample_iterations} + 1 slices.}
#'       \item{f}{The f parameter draws, stored in an array with \code{n} rows,
#'                \code{m} columns, and \code{sample_iterations} + 1 slices.}
#'       \item{IRFs}{Estimated item response functions for the items, with one
#'                   column per item, or \code{m} columns, and 1001 rows.
#'                   The first row has the probabilities of a 1 response
#'                   for a theta value of -5.0, the second the probability for
#'                   each item of a 1 response for a theta value of -4.99,
#'                   ..., and the last for a theta value of 5.0.}
#'   }
#'
#' @examples
#' ##### Monotonic IRT example ####
#' ## Simulate data
#' gen_responses <- function(theta, alpha, beta) {
#'     # Stardard 2PL model
#'     n <- length(theta)
#'     m <- length(alpha)
#'     responses <- matrix(0, n, m)
#'     for ( j in 1:m ) {
#'         for ( i in 1:n ) {
#'             p <- plogis(alpha[j] + beta[j] * theta[i])
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
#' samples <- gpirtMCMC(responses, 1, 0, vote_codes = list(yea = 1, nay = 0,
#'                                                         missing = NA))
#' str(samples)
#'
#' @export
gpirtMCMC <- function(data, sample_iterations, burn_iterations,
                      vote_codes = list(yea = 1:3, nay = 4:6,
                                        missing = c(0, 7:9, NA)),
                      beta_prior_means = matrix(0, nrow = 2, ncol = ncol(data)),
                      beta_prior_sds = matrix(3, nrow = 2, ncol = ncol(data)),
                      beta_proposal_sds = matrix(0.1, nrow = 2, ncol = ncol(data)),
                      theta_init = NULL, thresholds = NULL) {
    # First we make sure our data are in the proper format:
    if ( !is.null(vote_codes) ){
        data <- as.response_matrix(data, vote_codes)
    }
    
    # Now we make sure we have initial values for theta
    if ( is.null(theta_init) ) {
        theta_init  <- rnorm(nrow(data))
    }
    # Now we make sure we have initial values for thresholds
    if ( is.null(thresholds) ) {
        if(is.matrix(data)){
            unique_ys = unique(data)
        }else{
            unique_ys = unique(unlist(data))
        }   
        C = length(unique(unique_ys[!is.na(unique_ys)]))
        thresholds <- c(-Inf)
        for(i in 1:(C-1)){
            thresholds = c(thresholds, R::qnorm(i/C, 0, 1, 1, 0))
        }
        thresholds = c(thresholds, Inf)
    }
    # Now we can call the C++ sampler function
    result <- .gpirtMCMC(
        data, theta_init, sample_iterations, burn_iterations,
        beta_prior_means, beta_prior_sds, beta_proposal_sds, thresholds
    )
    # And return the result
    return(result)
}
