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
#'   coded as 1, an element named "nay" gives the repsonses that should be coded
#'   as -1, and an element named "missing" gives responses that should be NA;
#'   only used if \code{data} must be coerced to a \code{response_matrix} object
#' @param group An integer or character vector, or factor, of length
#'   \code{nrow(data)} identifying the group each respondent;
#'   only used if \code{data} must be coerced to a \code{response_matrix} object
#' @param prior_means A list of \code{length(group)} giving the prior mean
#'   for the ideology parameter for the respondents in each group or a vector
#'   of length \code{nrow(data)} giving the prior mean for each respondent's
#'   ideology parameter;
#'   only used if \code{data} must be coerced to a \code{response_matrix} object
#' @param sf A numeric vector of length one giving the scale factor for the
#'   covariance function for the Gaussian process prior; default is 1
#' @param ell A numeric vector of length one giving the length scale for the
#'   covariance function for the Gaussian process prior; default is 1
#' @param theta_init A vector of length \code{nrow(data)} giving initial values
#'   for the respondent ideology parameters; if NULL (the default), the initial
#'   values are drawn from the parameters' prior distributions.
#'
#' @return A list of length two; the first element, "theta", is a matrix of
#'   theta parameter draws (with dimensions (sample_iterations + 1) x n; the
#'   initial values are included), and the second element, "f", is an array of
#'   the f parameter draws (with dimensions n x m x (sample_iterations + 1)).
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
#' samples <- gpirtMCMC(responses, 1, 0, vote_codes = list(yea = 1, nay = 0,
#'                                                         missing = NA))
#' str(samples)
#'
#' @export
gpirtMCMC <- function(data, sample_iterations, burn_iterations,
                      vote_codes = list(yea = 1:3, nay = 4:6,
                                        missing = c(0, 7:9, NA)),
                      group = rep(0, nrow(data)),
                      prior_means = list(`0` = 0, `100` = -1, `200` = 1),
                      sf = 1, ell = 1, theta_init = NULL) {
    # First we make sure our data are in the proper format:
    data <- as.response_matrix(data, vote_codes, group, prior_means)
    # We'll convert prior_means and group to something more useful for things
    # on the C++ side
    if ( is.list(attr(data, "prior_means")) ) {
        mean_names <- names(attr(data, "prior_means"))
        groups <- match(attr(data, "group"), mean_names) - 1
        means  <- unlist(attr(data, "prior_means"))
    } else {
        means  <- unique(attr(data, "prior_means"))
        groups <- match(attr(data, "prior_means"), means) - 1
    }
    prior_means <- unlist(prior_means[as.character(group)])
    # Now we make sure we have initial values for theta
    if ( is.null(theta_init) ) {
        theta_init  <- rnorm(nrow(data), mean = prior_means)
    }
    # Now we can call the C++ sampler function
    result <- .gpirtMCMC(data, theta_init, sample_iterations, burn_iterations,
                         means, groups, sf^2, 1 / (ell^2))
    # And return the result
    return(result)
}
