#' Constrained joint maximum likelihood estimation for exploratory item factor analysis on the multidimensional two parameter logistic model.
#'
#' @param response N by J matrix containing 0/1/NA responses, where N is the number of respondents, J is the number of items, and NA indicates a missing response.
#' @param K The number of factors in exploratory item factor analysis.
#' @param theta0 N by K matrix, the initial value of latent factor scores for each respondent.
#' @param A0 J by K matrix, the initial value of loading matrix.
#' @param d0 Length J vector, the initial value of intercept parameters.
#' @param cc A constant constraining the magnitude of the norms of person and item parameter vectors.
#' @param tol The tolerance for convergence with a default value 1e-4.
#' @param print_proc Print the precision during the esitmation procedure with a default value TRUE.
#' @param parallel Whether or not enable the parallel computing with a default value FALSE.
#' 
#' @return The function returns a list with the following components:
#' \describe{
#'   \item{theta_hat}{The estimated person parameter matrix.}
#'   \item{A_hat}{The estimated loading parameter matrix}
#'   \item{d_hat}{The estimated intercept parameters.}
#' }
#' @references 
#' Chen, Y., Li, X., & Zhang, S. (2018). Joint Maximum Likelihood Estimation for High-Dimensional Exploratory Item Factor Analysis. \emph{Psychometrika}, 1-23. <doi:10.1007/s11336-018-9646-5>;
#' 
#' @examples
#' # load a simulated dataset
#' attach(data_sim)
#' 
#' # run the exploratory analysis
#' res <- mirtjml_expr(response, K)
#' 
#' 
#' @importFrom GPArotation GPFoblq
#' @export mirtjml_expr
mirtjml_expr <- function(response, K, theta0 = NULL, A0 = NULL, d0 = NULL, cc = NULL, 
                    tol = 1e-4, print_proc = TRUE, parallel = FALSE){
  N <- nrow(response)
  J <- ncol(response)
  nonmis_ind <- 1 - is.na(response)
  response[is.na(response)] <- 0
  if(is.null(theta0) || is.null(A0) || is.null(d0)){
    t1 <- Sys.time()
    if(print_proc){
      cat("\n", "Initializing... finding good starting point.\n")
    }
    initial_value = svd_start(response, nonmis_ind, K)
    t2 <- Sys.time()
  }
  if(is.null(theta0)){
    theta0 <- initial_value$theta0
  }
  if(is.null(A0)){
    A0 <- initial_value$A0
  }
  if(is.null(d0)){
    d0 <- initial_value$d0
  }
  if(is.null(cc)){
    cc = 5*sqrt(K)
  }
  res <- cjmle_expr_cpp(response, nonmis_ind, cbind(rep(1,N),theta0),
                        cbind(d0,A0), cc, tol, print_proc, parallel)
  res_standard <- standardization_cjmle(res$theta[,2:(K+1)], res$A[,2:(K+1)], res$A[,1])
  if(K > 1){
    temp <- GPFoblq(res_standard$A1, method = "geomin")
    A_rotated <- temp$loadings
    rotation_M <- temp$Th
    theta_rotated <- res_standard$theta1 %*% rotation_M
  } else{
    A_rotated <- res_standard$A1
    theta_rotated <- res_standard$theta1
  }
  t3 <- Sys.time()
  if(print_proc){
    cat("\n\n", "Precision reached!\n")
    cat("Time spent:\n")
    cat("Find start point: ", as.numeric(t2-t1)," second(s) | ", "Optimization: ", as.numeric(t3-t2)," second(s)\n")
  }
  return(list("theta_hat" = theta_rotated,
              "A_hat" = A_rotated,
              "d_hat" = res_standard$d1))
}


