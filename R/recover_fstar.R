#' Gaussian Process Recover IRT
#'
#' Recover fstar from f and seed
#'
#' @param seed An integer giving random seed.
#' @param f A matrix of latent function values of shape n by m by h, where
#'   n is number of respondents, m is number of items per session, and
#'   h is number of sessions.
#' @param y A matrix of shape n by m by h, giving responses for n respondents
#' IRFs and h sessions
#' @param theta A matrix of shape n by h, giving initial values
#'   for the respondent ideology parameters
#' @param beta A tensor of shape 2 by n by h, giving intercept/slope
#'   for the mean values of IRF
#' @param thresholds A vector of length C+1, giving threshold vector
#' @param beta_prior_means A numeric matrix of with \code{ncol(data)} columns
#'   and two rows giving the prior means for the items' linear means' intercept
#'   and slope; by default, a matrix of zeros
#' @param beta_prior_sds A numeric matrix of with \code{ncol(data)} columns
#'   and two rows giving the prior standard deviations for the items' linear
#'   means' intercept and slope; by default, a matrix of threes
#' @param constant_IRF A binary indicator of whether IRFs are constant across sessions
#'
#' @return A list with elements
#'   \describe{
#'       \item{fstar}{Estimated item response functions for the items, with one
#'                   column per item, or \code{m} columns, and 1001 rows.
#'                   The first row has the probabilities of a 1 response
#'                   for a theta value of -5.0, the second the probability for
#'                   each item of a 1 response for a theta value of -4.99,
#'                   ..., and the last for a theta value of 5.0.}
#'   }
#'
#'
#' @export 
recover_fstar <- function(seed, f, y, theta, beta, thresholds,
                      beta_prior_means = matrix(0, nrow = 2, ncol = ncol(y)),
                      beta_prior_sds = matrix(0.5, nrow = 2, ncol = ncol(y)),
                      constant_IRF=0) {
    # Now we can call the C++ function
    result <- .recover_fstar(
            seed, f, y, theta, beta, thresholds, beta_prior_means, beta_prior_sds, constant_IRF
    )
    # And return the result
    return(result)
}
