#' Create an object of class response_matrix
#'
#' \code{response_matrix} creates a response_matrix object, for use with
#' \code{\link{gpirtMCMC}}; \code{is.response_matrix} tests if its argument is
#' a \code{response_matrix} object; \code{as.response_matrix} attemps to turn
#' its argument into a \code{response_matrix}.
#'
#' @param data A matrix or dataframe of responses
#' @param yea_codes A vector giving the values corresponding to a "yea"
#'   response (default is 1)
#' @param nay_codes A vector giving the values corresponding to a "nay"
#'   response (default is 0)
#' @param missing_codes A vector giving the values corresponding to a missing
#'   response (default is NA)
#' @param x An R object
#'
#' @examples
#' ## Data in {0, 1}
#' x   <- c(1, 0, 1, 1, 0, NA)
#' ex1 <- matrix(x, nrow = 3)
#' ex2 <- data.frame(x1 = x[1:3], x2 = x[4:6])
#' response_matrix(ex1)
#' response_matrix(ex2)
#' ## Multiple "yea" codes
#' x   <- c(1, -1, 2, 3, -1, NA)
#' ex3 <- matrix(x, nrow = 3)
#' ex4 <- data.frame(x1 = x[1:3], x2 = x[4:6])
#' response_matrix(ex3, yea_codes = 1:3)
#' response_matrix(ex4, yea_codes = 1:3)
#' ## Dataframe with factors
#' ex5 <- data.frame(x = factor(c("Yea", "Nay", "Yea")),
#'                   y = factor(c("Yea", "Nay", NA)))
#' y <- response_matrix(ex5, yea_codes = "Yea", nay_codes = "Nay")
#' is.response_matrix(ex5)
#' is.response_matrix(y)
#' as.response_matrix(ex5, yea_codes = "Yea", nay_codes = "Nay")
#' as.response_matrix(y,   yea_codes = "Yea", nay_codes = "Nay")
#'
#' @export
response_matrix <- function(data, yea_codes = 1, nay_codes = 0,
                            missing_codes = NA) {
    # Lists that are not dataframes will cause problems
    if ( is.list(data) & !is.data.frame(data) ) {
        stop(paste("Conversion from lists to response_matrix objects",
                   "is currently unsupported."))
    }
    # Now we can coerce 'data' into a matrix;
    # we also need a copy in case the nay_code is 1 or yea_code is -1.
    result <- as.matrix(data)
    tmp    <- result
    # Next we fix the data so that yeas are 1 and nays are -1.
    result[which(tmp %in% yea_codes)]     <-  1
    result[which(tmp %in% nay_codes)]     <- -1
    result[which(tmp %in% missing_codes)] <- NA
    # If the input was a dataframe of factors, we'll have a character matrix
    # at this point, so this just guards against that
    result <- apply(result, 2, as.numeric)
    # Then we class and return the result
    class(result) <- "response_matrix"
    return(result)
}

# is.response_matrix() checks if an object 'x' is a response_matrix.
# Since S3 classes don't have any sanity checks, we add a few things here;
# we not only check for class name, but also that the object is a matrix,
# and that it only contains the values 1, -1, and NA. As long as those things
# are true, we should be just fine.

#' @rdname response_matrix
#' @export
is.response_matrix <- function(x) {
    return(
        "response_matrix" %in% class(x)
        & is.matrix(x)
        & all(x %in% c(NA, -1, 1))
    )
}

#' @rdname response_matrix
#' @export
as.response_matrix <- function(x, yea_codes = 1, nay_codes = 0,
                               missing_codes = NA) {
    if ( !is.response_matrix(x) ) {
        x <- response_matrix(x, yea_codes, nay_codes, missing_codes)
    }
    return(x)
}
