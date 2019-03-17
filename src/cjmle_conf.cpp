#include <RcppArmadillo.h>
#include "depend_funcs.h"
#include "mirtjml_omp.h"

using namespace std;
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::vec grad_neg_loglik_A_j_conf_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                                  const arma::vec &A_j, const arma::vec &Q_j, const arma::mat &theta){
  arma::vec tmp = response_j - 1 / (1 + exp(-theta * A_j));
  // arma::vec tmp1 = nonmis_ind_j % tmp;
  // arma::vec res = theta.row(0).t() * tmp1(0);
  // for(unsigned int i=1;i<theta.n_rows;++i){
  //   res += theta.row(i).t() * tmp1(i);
  // }
  return -(theta.t() * (nonmis_ind_j % tmp)) % Q_j;
  //return -res % Q_j;
}
// [[Rcpp::plugins(openmp)]]
arma::mat Update_A_conf_cpp(const arma::mat &A0, const arma::mat &Q, const arma::mat &response,
                       const arma::mat &nonmis_ind, const arma::mat &theta1, double cc, double step_A = 5){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = step_A;
    arma::vec h = grad_neg_loglik_A_j_conf_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), Q.row(j).t(), theta1);
    A1.col(j) = A0.row(j).t() - step * h;
    A1.col(j) = prox_func_cpp(A1.col(j), cc);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1.col(j), theta1) >
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1) &&
            step > 1e-4){
      step *= 0.5;
      A1.col(j) = A0.row(j).t() - step * h;
      A1.col(j) = prox_func_cpp(A1.col(j),cc);
    }
  }
  return(A1.t());
}
// [[Rcpp::plugins(openmp)]]
Rcpp::List Update_A_conf_init_cpp(const arma::mat &A0, const arma::mat &Q, const arma::mat &response,
                            const arma::mat &nonmis_ind, const arma::mat &theta1, double cc, double step_A = 100){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
  arma::vec final_step(J);
#pragma omp parallel for
  for(int j=0;j<J;++j){
    double step = step_A;
    arma::vec h = grad_neg_loglik_A_j_conf_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), Q.row(j).t(), theta1);
    A1.col(j) = A0.row(j).t() - step * h;
    A1.col(j) = prox_func_cpp(A1.col(j), cc);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1.col(j), theta1) >
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1) &&
            step > 1e-4){
      step *= 0.5;
      A1.col(j) = A0.row(j).t() - step * h;
      A1.col(j) = prox_func_cpp(A1.col(j),cc);
    }
    final_step(j) = step;
  }
  return(Rcpp::List::create(Rcpp::Named("A") = A1.t(),
                            Rcpp::Named("step_A") = final_step));
}
// [[Rcpp::export]]
Rcpp::List cjmle_conf_cpp(const arma::mat &response, const arma::mat &nonmis_ind, arma::mat theta0,
                                    arma::mat A0, arma::mat Q, double cc, double tol, bool print_proc, bool parallel){
  if(!parallel)
    omp_set_num_threads(1);
  else
    omp_set_num_threads(omp_get_num_procs());
  int N = theta0.n_rows;
  int J = A0.n_rows;
  int K = theta0.n_cols;
  // Adaptively find initial steps when updating A and theta
  Rcpp::List tmp_theta = Update_theta_init_cpp(theta0, response, nonmis_ind, A0, cc);
  arma::mat theta1 = tmp_theta[0];
  arma::vec theta_step = tmp_theta[1];
  double theta_init_step = mean(theta_step);
  Rcpp::List tmp_A = Update_A_conf_init_cpp(A0, Q, response, nonmis_ind, theta1, cc);
  arma::mat A1 = tmp_A[0];
  arma::vec A_step = tmp_A[1];
  double A_init_step = mean(A_step)/2;
  
  double eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
  while(eps > tol){
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, cc, theta_init_step);
    A1 = Update_A_conf_cpp(A0, Q, response, nonmis_ind, theta1, cc, A_init_step);
    eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
    // if(print_proc) Rprintf("\n eps: %f", eps);
    if(print_proc){
      double dist = (log(eps)-log(N*J)) / (log(tol)-log(N*J));
      Rcpp::Rcout<< "\r|";
      for(int i=0;i<floor(30*dist);++i){
        Rcpp::Rcout << "=";
      }
      for(int i=0;i<(30-floor(30*dist));++i){
        Rcpp::Rcout << " ";
      }
      int nn = ceil(100*dist);
      Rcpp::Rcout << "|" << min(100, nn) << "%, " << "eps: " << eps;
    }
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1.cols(1,K-1),
                            Rcpp::Named("d") = A1.cols(0,0),
                            Rcpp::Named("theta") = theta1.cols(1,K-1));
}
