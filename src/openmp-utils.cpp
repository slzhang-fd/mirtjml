#include "mirtjml_omp.h"
#ifdef _OPENMP
#include <pthread.h>
#endif
#include <errno.h>     // errno
#include <ctype.h>     // isspace

static int mirtjml_threads = 1;   

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

// [[Rcpp::export]]
int getmirtjml_threads() {
  // this is the main getter used by all parallel regions; they specify num_threads(getDTthreads())
  // Therefore keep it light, simple and robust. Local static variable. initDTthreads() ensures 1 <= DTthreads <= omp_get_num_proc()
  return mirtjml_threads;
}
// [[Rcpp::export]]
int setmirtjml_threads(int threads = -1) {
  // this is the main getter used by all parallel regions; they specify num_threads(getDTthreads())
  // Therefore keep it light, simple and robust. Local static variable. initDTthreads() ensures 1 <= DTthreads <= omp_get_num_proc()
#ifndef _OPENMP
  Rprintf("This installation of mirtjml has not been compiled with OpenMP support.\n");
#endif
  int old = mirtjml_threads;
  if(threads == -1){
    mirtjml_threads = omp_get_num_procs();
  } else{
    mirtjml_threads = imax(imin(omp_get_num_procs(), threads), 1);
  }
  // this output is captured, paste0(collapse="; ")'d, and placed at the end of test.data.table() for display in the last 13 lines of CRAN check logs
  // it is also printed at the start of test.data.table() so that we can trace any Killed events on CRAN before the end is reached
  // this is printed verbatim (e.g. without using data.table to format the output) in case there is a problem even with simple data.table creation/printing
  // Rprintf("  omp_get_num_procs()            %d\n", omp_get_num_procs());
  // Rprintf("  omp_get_thread_limit()         %d\n", omp_get_thread_limit());
  // Rprintf("  omp_get_max_threads()          %d\n", omp_get_max_threads());
  // Rprintf("  jsem is using %d threads. See ?setjsemthreads.\n", getjsem_threads());
  return old;
}