## Helper function for printing a concatenated list of (words, numbers, etc.)
## (to make pretty and informative message messages)
printc <- function(wlist) {
    n <- length(wlist)
    if ( n == 1 ) return(wlist)
    wlist[n] <- paste("and", wlist[n], collapse = " ")
    if ( n == 2 ) return(paste(wlist, collapse = " "))
    return(paste(wlist, collapse = ", "))
}

#' Create an object of class response_matrix
#'
#' \code{response_matrix} creates a response_matrix object, for use with
#' \code{\link{gpirtMCMC}}; \code{is.response_matrix} tests if its argument is
#' a \code{response_matrix} object; \code{as.response_matrix} attemps to turn
#' its argument into a \code{response_matrix}.
#'
#' @param data A matrix or dataframe of responses
#' @param response_codes A named list giving the mapping from recorded responses to
#'   {-1, 1, NA}. An element named "yea" gives the responses that should be
#'   coded as 1, an element named "nay" gives the repsonses that should be coded
#'   as -1, and an element named "missing" gives responses that should be NA
#' @param x An R object
#'
#' @examples
#' ## Data in {0, 1}
#' x   <- c(1, 0, 1, 1, 0, NA)
#' ex1 <- matrix(x, nrow = 3)
#' ex2 <- data.frame(x1 = x[1:3], x2 = x[4:6])
#' response_matrix(ex1, response_codes = list(yea = 1, nay = 0, missing = NA))
#' response_matrix(ex2, response_codes = list(yea = 1, nay = 0, missing = NA))
#' ## Multiple "yea" codes
#' x   <- c(1, -1, 2, 3, -1, NA)
#' ex3 <- matrix(x, nrow = 3)
#' ex4 <- data.frame(x1 = x[1:3], x2 = x[4:6])
#' response_matrix(ex3, response_codes = list(yea = 1:3, nay = -1, missing = NA))
#' response_matrix(ex4, response_codes = list(yea = 1:3, nay = -1, missing = NA))
#' ## Dataframe with factors
#' ex5 <- data.frame(x = factor(c("Yea", "Nay", "Yea")),
#'                   y = factor(c("Yea", "Nay", NA)))
#' y <- response_matrix(ex5, response_codes = list(yea = "Yea", nay = "Nay",
#'                                             missing = NA))
#' is.response_matrix(ex5)
#' is.response_matrix(y)
#' as.response_matrix(ex5, response_codes = list(yea = "Yea", nay = "Nay",
#'                                           missing = NA))
#' as.response_matrix(y,   response_codes = list(yea = "Yea", nay = "Nay",
#'                                           missing = NA))
#'
#' @export
response_matrix <- function(data,
                            response_codes = list(yea = 1:3, nay = 4:6,
                                                  missing = c(0, 7:9, NA))
                            ) {
    # Lists that are not dataframes will cause problems
    if ( is.list(data) & !is.data.frame(data) ) {
        stop(paste("Conversion from lists to response_matrix objects",
                   "is currently unsupported."))
    }
    # Now we can coerce 'data' into a matrix;
    # we also need a copy in case the nay_code is 1 or yea_code is -1.
    # We also want to make sure to preserve any row or column names
    # (note that for this we have to re-add rownames after the as.numeric()
    #  step below)
    rnames <- rownames(data)
    cnames <- colnames(data)
    result <- as.matrix(data)
    tmp    <- result
    colnames(result) <- cnames
    # Double check that we have no surprise values; if so, we treat them as
    # missing but give the user a message about it
    if ( !all(result %in% unlist(response_codes)) ) {
        omitted_responses <- setdiff(result, unlist(response_codes))
        response_codes$missing <- c(response_codes$missing, omitted_responses)
        message("Responses with value ", printc(omitted_responses), " were ",
                "not given a response code and will be treated as missing.")
    }
    # Next we fix the data so that yeas are 1 and nays are -1.
    result[which(tmp %in% response_codes$yea)]     <-  1
    result[which(tmp %in% response_codes$nay)]     <- -1
    result[which(tmp %in% response_codes$missing)] <- NA
    # If the input was a dataframe of factors, we'll have a character matrix
    # at this point, so this just guards against that
    result <- apply(result, 2, as.numeric)
    rownames(result) <- rnames
    # Now we guard against unanimity
    unanimous_items <- apply(result, 2, function(x) {
        length(unique(na.omit(x))) == 1
    })
    result <- result[ , !unanimous_items]
    if ( any(unanimous_items) ) {
        NU <- sum(unanimous_items)
        message("Item", "if"(NU > 1, "s ", " "), printc(which(unanimous_items)),
                "if"(NU > 1, " were", " was"), " discarded as unanimous.")
    }
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
as.response_matrix <- function(x,
                               response_codes = list(yea = 1:3, nay = 4:6,
                                                     missing = c(0, 7:9, NA))
                               ) {
    if ( !is.response_matrix(x) ) {
        x <- response_matrix(x, response_codes)
    }
    return(x)
}
