context("response_matrix")

test_that("Does is.response_matrix function properly", {
    all_true            <- matrix(1)
    class(all_true)     <- "response_matrix"
    class_false         <- matrix(1)
    matrix_false        <- 1
    class(matrix_false) <- "response_matrix"
    values_wrong        <- matrix(6)
    class(values_wrong) <- "response_matrix"
    expect_false(is.response_matrix(class_false))
    expect_false(is.response_matrix(matrix_false))
    expect_false(is.response_matrix(values_wrong))
    expect_true(is.response_matrix(all_true))
})
