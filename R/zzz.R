.onLoad <- function(libname, pkgname) {
  setMIRTthreads()
}
#' @importFrom utils packageVersion 
.onAttach <- function(libname, pkgname){
  v = packageVersion("mirtjml")
  packageStartupMessage("mirtjml ", v, " using ", getMIRTthreads(), " threads (see ?getMIRTthreads())")
  if (!ChasOpenMP())
    packageStartupMessage("**********\nThis installation of data.table has not detected OpenMP support. It should still work but in single-threaded mode.",
                          " If this is a Mac, please ensure you are using R>=3.4.0 and have followed our Mac instructions here: https://github.com/slzhang-fd/mirtjml/wiki/Installation.",
                          " This warning message should not occur on Windows or Linux. If it does, please file a GitHub issue.\n**********")
}