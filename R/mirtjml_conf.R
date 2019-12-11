#' Constrained joint maximum likelihood estimation for confirmatory item factor analysis on the multidimensional two parameter logistic model.
#'
#' @param response N by J matrix containing 0/1/NA responses, where N is the number of respondents, J is the number of items, and NA indicates a missing response.
#' @param Q J by K matrix containing 0/1 entries, where J is the number of items and K is the number of latent traits. Each entry indicates whether an item measures a certain latent trait.
#' @param theta0 N by K matrix, the initial value of latent factor scores for each respondent.
#' @param A0 J by K matrix, the initial value of loading matrix, satisfying the constraints given by Q.
#' @param d0 Length J vector, the initial value of intercept parameters.
#' @param cc A constant constraining the magnitude of the norms of person and item parameter vectors.
#' @param tol The tolerance for convergence with a default value 5.
#' @param print_proc Print the precision during the esitmation procedure with a default value TRUE.
#' 
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{theta_hat}{The estimated person parameter matrix.}
#'   \item{A_hat}{The estimated loading parameter matrix}
#'   \item{d_hat}{The estimated intercept parameters.}
#' }
#' @references 
#' Chen, Y., Li, X., & Zhang, S. (2019). Structured Latent Factor Analysis for Large-scale Data: Identifiability, Estimability, and Their Implications. 
#' \emph{Journal of the American Statistical Association} <doi: 10.1080/01621459.2019.1635485>.
#' @examples
#' # load a simulated dataset
#' attach(data_sim)
#' 
#' # generate starting values for the algorithm
#' A0 <- Q
#' d0 <- rep(0, J)
#' theta0 <- matrix(rnorm(N*K, 0, 1),N)
#' 
#' # use all available cores by running
#' # setMIRTthreads(-1)
#' 
#' # run the confirmatory analysis
#' res_conf <- mirtjml_conf(response, Q, theta0, A0, d0)
#' 
#' 
#' @importFrom stats cov
#' @export mirtjml_conf
mirtjml_conf <- function(response, Q, theta0, A0, d0, cc = NULL, tol = 5, print_proc = TRUE){
  N <- nrow(response)
  J <- ncol(response)
  K <- ncol(Q)
  nonmis_ind <- 1 - is.na(response)
  response[is.na(response)] <- 0
  if(is.null(cc)){
    cc = 5*sqrt(K)
  }
  t1 <- Sys.time()
  res <- cjmle_conf_cpp(response, nonmis_ind, cbind(rep(1,N),theta0),cbind(d0,A0), 
                        cbind(rep(1,J),Q), cc, tol, print_proc)
  t2 <- Sys.time()
  if(print_proc){
    cat("\n\n", "Precision reached!\n")
    cat("Time spent:  ", as.numeric(t2-t1), " second(s).\n")
  }
  # scaling
  tmp <- sqrt(diag(cov(res$theta)))
  return(list("theta_hat" = res$theta %*% diag(1/tmp),
              "A_hat" = res$A %*% diag(tmp),
              "d_hat" = res$d))
}


