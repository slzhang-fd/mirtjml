.onLoad <- function(libname, pkgname) {
  setMIRTthreads()
}
#' @importFrom utils packageVersion 
.onAttach <- function(libname, pkgname){
  v = packageVersion("mirtjml")
  packageStartupMessage("mirtjml ", v, " using ", getMIRTthreads(), " threads (see ?getMIRTthreads())")
}