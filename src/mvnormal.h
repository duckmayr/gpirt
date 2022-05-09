#ifndef GPIRT_MVNORMAL_H
#define GPIRT_MVNORMAL_H

inline arma::vec rmvnorm(const arma::mat& cholS) {
    arma::uword m = cholS.n_cols;
    arma::vec res(m);
    for (arma::uword i = 0; i < m; ++i ) {
        res[i] = R::rnorm(0.0, 1.0);
    }
    return cholS * res;
}

#endif

