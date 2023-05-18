clean_responses = function(responses) {
    unanimous = apply(responses, 2, function(x) length(unique(na.omit(x))) == 1)
    return(responses[ , !unanimous])
}

#' Format Responses for the Dynamic, Ordinal GPIRT Model
#'
#' Turn a dataframe of responses from multiple time periods into a list of
#' matrices for [gpirt::gpirtMCMC()]
#'
#' @param dataframe A data.frame containing observed responses
#' @param respondent_ids The column indicating which respondent an observation
#'     (i.e. dataframe row) is for
#' @param item_ids The column indicating which item an observation
#'     (i.e. dataframe row) is for
#' @param period_ids The column indicating which time period an observation
#'     (i.e. dataframe row) is for
#' @param responses The column recording the observed response
#'
#' @return A list with the same number of unique elements as
#'     `dataframe$period_ids`, where each element is a matrix of the observed
#'     responses in a given time period. The rows of each matrix correspond to a
#'     respondent, the columns of each matrix correspond to an item, and the
#'     elements of the list correspond to a time period. The elments of the list
#'     will be named by the time period IDs, the rows of each matrix will be
#'     named by the respondent IDs, and the columns of each matrix will be named
#'     by the item IDs.
#'
#' @export
format_dogpirt_responses = function(
        dataframe,
        respondent_ids,
        item_ids,
        period_ids,
        responses
    ) {
    ## We use dplyr & tidyr for data munging; first ensure they're installed
    dependencies = c("dplyr", "tidyr")
    has_dependency = sapply(dependencies, require, character.only = TRUE)
    if ( any(!has_dependency) ) {
        needed = dependencies[!has_dependency]
        stop("Please install ", paste(needed, collapse = " and "))
    }
    ## Set up the list for response matrices
    periods = dataframe %>% pull({{period_ids}}) %>% unique() %>% sort()
    result  = vector(mode = "list", length = length(periods))
    ## Then for each time period
    for ( k in seq_along(periods) ) {
        ## Get the responses from just that period, in wide format
        tmp = dataframe %>%
            filter({{period_ids}} == periods[k]) %>%
            select({{respondent_ids}}, {{item_ids}}, {{responses}}) %>%
            pivot_wider(names_from = {{item_ids}}, values_from = {{responses}})
        ## Turn this into a matrix with respondent IDs as rownames,
        ## making sure to eliminate unanimous items
        rnames = tmp %>% pull({{respondent_ids}})
        tmp = tmp %>% select(-{{respondent_ids}})
        tmp = tmp %>% as.matrix %>% clean_responses
        rownames(tmp) = rnames
        result[[k]] = tmp
    }
    ## Name and return the results
    names(result) = periods
    return(result)
}
