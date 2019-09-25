.onLoad <- function(libname, pkgname) {
  setmirtjml_threads(-1)
}

.onAttach <- function(libname, pkgname){
  v = packageVersion("mirtjml")
  packageStartupMessage("mirtjml ", v, " using ", getmirtjml_threads(), " threads (see ?getmirtjml_threads())")
}