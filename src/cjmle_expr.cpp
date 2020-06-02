#include <RcppArmadillo.h>
#include "depend_funcs.h"
#include "mirtjml_omp.h"

//' @useDynLib mirtjml
//' @importFrom Rcpp evalCpp

using namespace std;
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
arma::vec grad_neg_loglik_A_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                                  const arma::vec &A_j, const arma::mat &theta){
  arma::vec tmp = response_j - 1.0 / (1.0 + arma::exp(-theta * A_j));
  return -theta.t() * (nonmis_ind_j % tmp);
}
// [[Rcpp::plugins(openmp)]]
arma::mat Update_A_cpp(const arma::mat &A0, const arma::mat &response, const arma::mat &nonmis_ind,
                       const arma::mat &theta1, double cc, double step_A = 5){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
#pragma omp parallel for num_threads(getmirtjml_threads())
  for(int j=0;j<J;++j){
    double step = step_A; 
    arma::vec h = grad_neg_loglik_A_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1);
    A1.col(j) = A0.row(j).t() - step * h;
    A1.col(j) = prox_func_cpp(A1.col(j), cc);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1.col(j), theta1) > 
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1) &&
          step > 1e-7){
      step *= 0.5;
      A1.col(j) = A0.row(j).t() - step * h;
      A1.col(j) = prox_func_cpp(A1.col(j), cc);
    }
    //Rcpp::Rcout << "\n final step loop when updating A = "<< -log(step/step_A)/log(2)<< "\n";
  }
  return(A1.t());
}
// [[Rcpp::plugins(openmp)]]
Rcpp::List Update_A_init_cpp(const arma::mat &A0, const arma::mat &response, const arma::mat &nonmis_ind,
                       const arma::mat &theta1, double cc, double step_A = 100){
  arma::mat A1 = A0.t();
  int J = A0.n_rows;
  arma::vec final_step(J);
#pragma omp parallel for num_threads(getmirtjml_threads())
  for(int j=0;j<J;++j){
    double step = step_A; 
    arma::vec h = grad_neg_loglik_A_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1);
    A1.col(j) = A0.row(j).t() - step * h;
    A1.col(j) = prox_func_cpp(A1.col(j), cc);
    while(neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A1.col(j), theta1) > 
            neg_loglik_j_cpp(response.col(j), nonmis_ind.col(j), A0.row(j).t(), theta1) &&
            step > 1e-7){
      step *= 0.5;
      A1.col(j) = A0.row(j).t() - step * h;
      A1.col(j) = prox_func_cpp(A1.col(j), cc);
    }
    final_step(j) = step;
  }
  return(Rcpp::List::create(Rcpp::Named("A") = A1.t(),
                            Rcpp::Named("step_A") = final_step));
}
// [[Rcpp::export]]
Rcpp::List cjmle_expr_cpp(const arma::mat &response, const arma::mat &nonmis_ind, arma::mat theta0,
                                arma::mat A0, double cc, double tol, bool print_proc){
  int N = theta0.n_rows;
  int J = A0.n_rows;
  // Adaptively find initial steps when updating A and theta
  Rcpp::List tmp_theta = Update_theta_init_cpp(theta0, response, nonmis_ind, A0, cc);
  arma::mat theta1 = tmp_theta[0];
  arma::vec theta_step = tmp_theta[1];
  double theta_init_step = mean(theta_step);
  Rcpp::List tmp_A = Update_A_init_cpp(A0, response, nonmis_ind, theta1, cc);
  arma::mat A1 = tmp_A[0];
  arma::vec A_step = tmp_A[1];
  double A_init_step = mean(A_step)/2;
  
  double eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
  while(eps > tol){
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, cc, theta_init_step);
    A1 = Update_A_cpp(A0, response, nonmis_ind, theta1, cc, A_init_step);
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
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("obj") = neg_loglik(theta1*A1.t(), response, nonmis_ind));
}
//' Simulation for the book chapter
Rcpp::List cjmle_expr_simu(const arma::mat &response, const arma::mat &nonmis_ind, arma::mat theta0,
                           arma::mat A0, arma::mat A_true, double cc, double tol, int max_steps = 1000,
                           bool print_proc = false){
  int K = A0.n_cols - 1;
  
  int mcn = 0;
  // clock_t begin = clock(); 
  // double elapsed_secs0, elapsed_secs = 0;
  // int time_record_num = (int)max_time / 1;
  // arma::vec time_points = arma::zeros(time_record_num);
  arma::vec sin_angle = arma::zeros(max_steps);
  // arma::vec step_points = arma::zeros(time_record_num);
  
  // obtain left sigular vectors for true A
  arma::mat U1, V1, U2, V2;
  arma::vec s1, s2;
  arma::svd(U1, s1, V1, A_true);
  U1 = U1.cols(0, K-1);
  
  // Adaptively find initial steps when updating A and theta
  Rcpp::List tmp_theta = Update_theta_init_cpp(theta0, response, nonmis_ind, A0, cc);
  arma::mat theta1 = tmp_theta[0];
  arma::vec theta_step = tmp_theta[1];
  double theta_init_step = arma::mean(theta_step);
  Rcpp::List tmp_A = Update_A_init_cpp(A0, response, nonmis_ind, theta1, cc);
  arma::mat A1 = tmp_A[0];
  arma::vec A_step = tmp_A[1];
  double A_init_step = arma::mean(A_step)/2;
  
  double eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
  while(eps > tol){
    mcn += 1;
    theta0 = theta1;
    A0 = A1;
    theta1 = Update_theta_cpp(theta0, response, nonmis_ind, A0, cc, theta_init_step);
    A1 = Update_A_cpp(A0, response, nonmis_ind, theta1, cc, A_init_step);
    eps = neg_loglik(theta0*A0.t(), response, nonmis_ind) - neg_loglik(theta1*A1.t(), response, nonmis_ind);
    
    // calculate cosine between A_hat and A_true
    arma::svd(U2, s2, V2, A1.cols(1, K));
    arma::vec s_tmp = arma::svd(U1.t() * U2.cols(0, K-1));
    
    // clock_t end = clock();
    // elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // if(elapsed_secs - elapsed_secs0 > 1){
    //   Rcpp::Rcout << "eps: " << eps << " elapsed time: " << elapsed_secs <<
    //     " step: " << mcn <<
    //     " angle A_true, A_hat: " << std::sqrt(1 - s_tmp(K-1)) << std::endl;
    //   time_ind = (int)elapsed_secs / 1;
    //   time_points[time_ind-1] = elapsed_secs;
    //   sin_angle[time_ind -1] = std::sqrt(1 - s_tmp(K-1));
    //   step_points[time_ind - 1] = mcn;
    //   elapsed_secs0 = elapsed_secs;
    // }
    sin_angle(mcn -1) = std::sqrt(1 - s_tmp(K-1));
    if(print_proc)
      Rcpp::Rcout << "eps: " << eps << " step: " << mcn <<
        " angle A_true, A_hat: " << std::sqrt(1 - s_tmp(K-1)) << std::endl;
  }
  return Rcpp::List::create(Rcpp::Named("A") = A1,
                            Rcpp::Named("theta") = theta1,
                            Rcpp::Named("obj") = neg_loglik(theta1*A1.t(), response, nonmis_ind),
                            Rcpp::Named("sin_angle") = sin_angle.subvec(0, mcn-1));
}

