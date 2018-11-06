// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::vec prox_func_cpp(const arma::vec &y, double C){
  double y_norm2 = arma::accu(square(y));
  if(y_norm2 <= C*C){
    return y;
  }
  else{
    return sqrt(C*C / y_norm2) * y;
  }
}
arma::vec prox_func_theta_cpp(arma::vec y, double C){
  double y_norm2 = arma::accu(square(y)) - 1;
  if(y_norm2 <= C*C-1){
    return y;
  }
  else{
    y = sqrt((C*C-1) / y_norm2) * y;
    y(0) = 1;
    return y;
  }
}

double neg_loglik(const arma::mat &thetaA, const arma::mat &response, const arma::mat &nonmis_ind){
  int N = response.n_rows;
  int J = response.n_cols;
  double res = arma::accu( nonmis_ind % (thetaA % response - log(1+exp(thetaA))) );
  return -res / N / J;
}

double neg_loglik_i_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                        const arma::mat &A, const arma::vec &theta_i){
  //int J = response_i.n_elem;
  arma::vec tmp = A * theta_i;
  return -arma::accu(nonmis_ind_i % (tmp % response_i - log(1 + exp(tmp))));
}

arma::vec grad_neg_loglik_thetai_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                                     const arma::mat &A, const arma::vec &theta_i){
  //int J = response_i.n_elem;
  arma::vec tmp = response_i - 1 / (1 + exp(- A * theta_i));
  arma::mat tmp1 = -arma::diagmat(nonmis_ind_i % tmp) * A;
  return tmp1.t() * arma::ones(response_i.n_rows);
}

// [[Rcpp::plugins(openmp)]]
arma::mat Update_theta_cpp(const arma::mat &theta0, const arma::mat &response,
                           const arma::mat &nonmis_ind, const arma::mat &A0, double C){
  arma::mat theta1 = theta0.t();
  int N = response.n_rows;
#pragma omp parallel for
  for(int i=0;i<N;++i){
    double step = 1;
    arma::vec h = grad_neg_loglik_thetai_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t());
    h(0) = 0;
    theta1.col(i) = theta0.row(i).t() - step * h;
    theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
    while(neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta1.col(i)) >
            neg_loglik_i_cpp(response.row(i).t(), nonmis_ind.row(i).t(), A0, theta0.row(i).t()) &&
            step > 1e-4){
      step *= 0.5;
      theta1.col(i) = theta0.row(i).t() - step * h;
      theta1.col(i) = prox_func_theta_cpp(theta1.col(i), C);
      if(step <= 1e-4){
        Rprintf("error in update theta\n");
        //theta1.col(i) = theta0.row(i).t();
      }
    }
  }
  return(theta1.t());
}

double neg_loglik_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                        const arma::vec &A_j, const arma::mat &theta){
  //int N = response_j.n_elem;
  arma::vec tmp = theta * A_j;
  return -arma::accu(nonmis_ind_j % (tmp % response_j - log(1+exp(tmp))));
}
