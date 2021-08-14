library(ggplot2)
library(hexSticker)
# ?sticker
covSEiso <- function(x, z = x) {
    n <- length(x)
    m <- length(z)
    K <- matrix(nrow = n, ncol = m)
    for ( j in 1:m ) {
        for ( i in 1:n ) {
            K[i, j] <- exp(-0.5 * (x[i] - z[j])^2)
        }
    }
    return(K)
}
set.seed(714)
n <- 100
theta <- seq(from = -2, to = 2, length.out = n)
K <- covSEiso(theta)
f <- theta + c(t(mvtnorm::rmvnorm(n = 1, sigma = K)))
p <- plogis(f)
h <- theta^2 + c(t(mvtnorm::rmvnorm(n = 1, sigma = K))) - 1
q <- plogis(h)
d <- data.frame(theta = theta, p = p, q = q)
g <- ggplot(data = d, mapping = aes(x = theta, y = p)) +
    geom_line(group = 1) +
    geom_line(aes(y = q), group = 1, linetype = "dashed") +
    # ylim(0, 1) +
    scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1)) +
    xlab(expression(theta)) +
    ylab("P(y = 1)") +
    theme_classic() +
    theme_transparent() +
    theme(
        axis.text = element_text(colour = "black", size = 18),
        axis.title = element_text(size = 18),
        axis.ticks = element_blank()
    )
sticker(
    subplot = g, s_width = 1.5, s_height = 1, s_x = 0.95,
    package = "gpirt", p_size = 20, p_fontface = "bold", p_y = 1.5,
    h_fill = "#729fcf", h_color = "#204a87"
)
