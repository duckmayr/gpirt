#' Plot Item Response Functions
#' 
#' @param gp_results A list returned from \code{\link[gpmlr]{gp}}
#'   (see details and examples).
#' @param theta_star A numeric vector of theta values to predict the
#'   probability of a 1 response at.
#' @param main_title (Optional) A title for the plot
#' @param rug A logical vector of length one; if \code{TRUE} a rug of the
#'   theta estimates whose responses were 1 will be placed at the top of
#'   the plot and a rug of the theta estimates for respondents whose
#'   responses were -1 will be placed at the bottom of the plot
#'   (default is \code{FALSE})
#' @param responses An integer vector of observed responses;
#'   must be supplied if \code{rug} is \code{TRUE}
#' @param estimates A numeric vector of theta estimates for your respondents;
#'   must be supplied if \code{rug} is \code{TRUE}
#' 
#' @examples
#' \donttest{
#'     library(gpmlr)
#'     set.seed(123)
#'     theta <- rnorm(100)
#'     f <- 3 * sin(theta)
#'     P <- 1 / (1 + exp(-f))
#'     responses <- sapply(P, function(p) sample(c(1, -1), 1, prob = c(p, 1-p)))
#'     theta_star <- seq(-3, 3, 0.01)
#'     hyperparameters <- list(mean = numeric(), cov = c(0, 0))
#'     gp_results <- gp(hyperparameters, "infLaplace", "meanZero",
#'                      "covSEiso", "likLogistic", theta, responses,
#'                      theta_star)
#'     plot_irf(gp_results, theta_star, rug = TRUE, responses = responses,
#'              estimates = theta)
#'     z <- 3 * sin(theta_star)
#'     p <- 1 / (1 + exp(-z))
#'     lines(theta_star, p, lty = 2)
#' }
#' 
#' @export
plot_irf <- function(gp_results, theta_star, main_title = "",
                     rug = FALSE, responses = NULL, estimates = NULL) {
    ilogit <- function(x) 1 / (1 + exp(-x))
    margins <- "if"(main_title != "", c(3, 3, 3, 1), c(3, 3, 1, 1)) + 0.1
    opar <- graphics::par(mar = margins)
    on.exit(graphics::par(opar))
    fmu <- gp_results$FMU
    fsd <- sqrt(gp_results$FS2)
    low <- ilogit(fmu - 1.96 * fsd)
    hi  <- ilogit(fmu + 1.96 * fsd)
    graphics::plot(theta_star, fmu, type = "n", ylim = 0:1, main = main_title,
                   xaxt = "n", yaxt = "n", xlab = "", ylab = "")
    xat <- seq(from = floor(min(theta_star)), to = ceiling(max(theta_star)))
    yat <- seq(from = 0, to = 1, by = 0.25)
    graphics::axis(side = 1, at = xat, tick = FALSE, line = -0.75)
    graphics::axis(side = 2, at = yat, tick = FALSE, line = -0.75,
                   labels = sprintf("%0.2f", yat))
    graphics::mtext(text = bquote(theta), side = 1, line = 1.5)
    graphics::mtext(text = bquote("Pr("~y[ij]~"= 1)"), side = 2, line = 1.5)
    graphics::polygon(x = c(theta_star, rev(theta_star)),
                      y = c(low, rev(hi)), border = NA, col = "#0072b240")
    graphics::lines(theta_star, ilogit(fmu), col = "#0072b2")
    if ( rug ) {
        if ( is.null(responses) ) {
            stop("Responses needed for rugs.")
        } else if ( is.null(estimates) ) {
            stop("Latent trait estimates needed for rugs.")
        }
        graphics::rug(x = estimates[responses ==  1], side = 3)
        graphics::rug(x = estimates[responses == -1], side = 1)
    }
}
