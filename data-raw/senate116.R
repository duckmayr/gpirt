## code to prepare `senate116` dataset goes here
## The assumption is that you're doing this from the main project directory
## The Senate data was downloaded from Voteview.com on 2020/01/31
## We'll use the full first session for the example data
senate116 <- read.csv("data-raw/S116_votes.csv", stringsAsFactors = FALSE)
rollcalls <- read.csv("data-raw/S116_rollcalls.csv", stringsAsFactors = FALSE)
session1  <- rollcalls$rollnumber[rollcalls$session == 1]
senate116 <- senate116[which(senate116$rollnumber %in% session1), ]
usethis::use_data(senate116)
