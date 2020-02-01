#' Rollcall votes in the U.S. Senate's first session of the 116th Congress
#'
#' @format A data frame with 42800 rows and 6 variables:
#' \describe{
#'   \item{congress}{Congress the vote occurred in; in this dataset, only 116}
#'   \item{chamber}{Chamber the vote occurred in; in this dataset, only "Senate"}
#'   \item{rollnumber}{ID number for the rollcall votes}
#'   \item{icpsr}{ID number for the Senators}
#'   \item{cast_code}{Integer indicators of how the Senators voted;
#'                    1 indicates a "Yea" vote, 6 indicates a "Nay" vote,
#'                    7 indicates a "Present" vote, and 9 indicates abstention.}
#'   \item{prob}{Probability of the observed vote according to NOMINATE}
#' }
#' @source \url{https://voteview.com/data}
"senate116"
