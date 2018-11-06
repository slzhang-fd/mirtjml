standardization_cjmle <- function(theta0, A0, d0){
  theta0 <- as.matrix(theta0)
  A0 <- as.matrix(A0)
  N <- nrow(theta0)
  d1 <- d0 + 1/N * A0 %*% t(theta0) %*% matrix(rep(1,N),N)
  svd_res = svd(theta0 - 1/N * matrix(rep(1,N),N) %*% matrix(rep(1,N),1) %*% theta0)
  theta1 = svd_res$u * sqrt(N)
  A1 = 1/sqrt(N) * A0 %*% svd_res$v %*% diag(svd_res$d, nrow = length(svd_res$d), ncol = length(svd_res$d))
  return(list("theta1"=theta1, "A1"=A1, "d1"=d1))
}
svd_start <- function(response, nonmis_ind, K, tol = 0.01){
  N <- nrow(response)
  J <- ncol(response)
  p_hat <- sum(nonmis_ind) / N / J
  X <- (2 * response - 1) * nonmis_ind
  temp <- svd(X)
  eff_num <- sum(temp$d >= 2*sqrt(N*p_hat))
  diagmat <- matrix(0, eff_num, eff_num)
  diag(diagmat) <- temp$d[1:eff_num]
  X <- as.matrix(temp$u[,1:eff_num]) %*% diagmat %*% t(as.matrix(temp$v[,1:eff_num]))
  X <- (X+1) / 2
  X[X<tol/2] <- tol
  X[X>(1-tol/2)] <- 1-tol
  M <- log(X / (1 - X))
  d0 <- colMeans(M)
  temp <- svd(t(t(M) - d0))
  theta0 <- sqrt(N) * as.matrix(temp$u[,1:K])
  A0 <- 1 / sqrt(N) * as.matrix(temp$v[,1:K]) %*% diag(temp$d[1:K], nrow = K, ncol = K)
  return(list("theta0"=theta0, "A0"=A0, "d0"=d0))
}
