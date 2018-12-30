#ifndef __DEPEND_FUNCS__
#define __DEPEND_FUNCS__
#include <RcppArmadillo.h>

arma::vec prox_func_cpp(const arma::vec &y, double C);
arma::vec prox_func_theta_cpp(arma::vec y, double C);
double neg_loglik(const arma::mat &thetaA, const arma::mat &response, const arma::mat &nonmis_ind);
double neg_loglik_i_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                        const arma::mat &A, const arma::vec &theta_i);
arma::vec grad_neg_loglik_thetai_cpp(const arma::vec &response_i, const arma::vec &nonmis_ind_i,
                                     const arma::mat &A, const arma::vec &theta_i);
arma::mat Update_theta_cpp(const arma::mat &theta0, const arma::mat &response,
                           const arma::mat &nonmis_ind, const arma::mat &A0, double C, double step_theta=200);
Rcpp::List Update_theta_init_cpp(const arma::mat &theta0, const arma::mat &response,
                                 const arma::mat &nonmis_ind, const arma::mat &A0, double C, double step_theta=1000);
double neg_loglik_j_cpp(const arma::vec &response_j, const arma::vec &nonmis_ind_j,
                        const arma::vec &A_j, const arma::mat &theta);
#endif
