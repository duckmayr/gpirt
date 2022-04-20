context("response_matrix")

#
# Specify objects needed for tests.
#
x   <- c(1, 0, 1, 1, 0, NA)
ex1 <- matrix(x, nrow = 3)
ex2 <- data.frame(x1 = x[1:3], x2 = x[4:6])
result1 <- response_matrix(ex1, response_codes = list(yea = 1, nay = 0,
                                                      missing = NA))
result2 <- response_matrix(ex2, response_codes = list(yea = 1, nay = 0,
                                                      missing = NA))
## Multiple "yea" codes
x   <- c(1, -1, 2, 3, -1, NA)
ex3 <- matrix(x, nrow = 3)
ex4 <- data.frame(x1 = x[1:3], x2 = x[4:6])
result3 <- response_matrix(ex3, response_codes = list(yea = 1:3, nay = -1,
                                                      missing = NA))
result4 <- response_matrix(ex4, response_codes = list(yea = 1:3, nay = -1,
                                                      missing = NA))
## Dataframe with factors
ex5 <- data.frame(x = factor(c("Yea", "Nay", "Yea")),
                  y = factor(c("Yea", "Nay", NA)))
result5 <- response_matrix(ex5, response_codes = list(yea = "Yea", nay = "Nay",
                                                      missing = NA))
ex6 <- data.frame(x = factor(c("Yea", "Nay", "Yes")),
                  y = factor(c("Yea", "Nay", NA)))

# 06.07.2019: The following set of comments need to be updated at some point and do not
# reflect the current functionality of the tests.
#
# Some responsibility must be put on the user to properly format data.
#
# For now, let's assume that the input for response_matrix MUST be a non-empty data frame.
#
# Also assume that data is binary with the potential for a (but not a required) missing data code.
#   Consequently, don't error out if there is only one entry type, but do throw a warning to the user
#   that this might be weird.
#   If the types of entries are > 3, report counts of frequencies by entry type. Give the option to the
#   user to print all observation indices matching a specific entry. Allow the user to keep doing this
#   until the user chooses an exit option for the presented counts. This will potentially help the user
#   identify stray codes in a large data set that they can fix themselves.

# Unit tests for response_matrix will require input to be:
#   (1) a data frame,
#   (2) a non-empty data frame,
#   (3) a dataframe containing entries with at most 3 different types of codes,
#   IF yea_codes, nay_codes, AND missing_codes ARE *NOT* SPECIFIED BY THE USER:
#       (4) entries must be from {1,0,NA}
#   ELSE:
#       (5) yea_codes != nay_codes != missing_codes
#       (6) entries must be from {yea_codes, nay_codes, missing_codes}
test_that("response_matrix functions properly", {
    expect_s3_class(result1, "response_matrix")
    expect_setequal(c(result1), c(1,-1, NA))
    expect_s3_class(result2, "response_matrix")
    expect_setequal(c(result2), c(1,-1, NA))
    expect_s3_class(result3, "response_matrix")
    expect_setequal(c(result3), c(1,-1, NA))
    expect_s3_class(result4, "response_matrix")
    expect_setequal(c(result4), c(1,-1, NA))
    expect_s3_class(result5, "response_matrix")
    expect_setequal(c(result5), c(1,-1, NA))
    expect_message(response_matrix(ex6, response_codes = list(yea = "Yea",
                                                              nay = "Nay",
                                                              missing = NA)))
    expect_error(response_matrix(list(1)), "Conversion from lists")
})


#
# Add documentation of unit testing goals for is.response_matrix.
#
test_that("is.response_matrix functions properly", {
    all_true            <- matrix(1)
    class(all_true)     <- "response_matrix"
    matrix_false        <- 1
    class(matrix_false) <- "response_matrix"
    values_wrong        <- matrix(6)
    class(values_wrong) <- "response_matrix"
    class_false         <- matrix(1)
    expect_false(is.response_matrix(class_false))
    expect_false(is.response_matrix(matrix_false))
    expect_false(is.response_matrix(values_wrong))
    expect_true(is.response_matrix(all_true))
})

#
# Add documentation of unit testing goals for as.response_matrix.
#
test_that("as.response_matrix functions properly", {
    expect_identical(as.response_matrix(ex1,
                                        response_codes = list(yea = 1, nay = 0,
                                                              missing = NA)),
                     result1)
    expect_identical(as.response_matrix(result1,
                                        response_codes = list(yea = 1, nay = 0,
                                                              missing = NA)),
                     result1)
})

