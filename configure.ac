## My approach here was inspired and informed by RcppArmadillo's configure.ac,
## though any shared code should be minimal; I rewrote the file to make sure
## (1) that I understood exactly what needed to be checked for, and
## (2) that I didn't have any superfluous code

AC_INIT(gpirt, 0.0.0.9000)


## Set R_HOME (see Writing R Extensions, section 1.2; this is taken from there)

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi


## Make sure we can handle C++

AC_LANG(C++)
AC_REQUIRE_CPP


## Now we need to set the OpenMP flags.

AC_MSG_CHECKING([for OpenMP])
openmp_flag=""

## We can use OpenMP if:
##     1. On Mac, we have clang >= 3.8
##     2. On other systems, we have g++ >= 5.4
## So, we collect the system name, whether the compiler version mentions
## g++/gcc, Apple LLVM, and clang, and  the major and minor version numbers

sys_name=$("${R_HOME}/bin/Rscript" --vanilla -e 'cat(Sys.info()[["sysname"]])')
cxx_version=$(${CXX} --version)
v_no=$(echo "$cxx_version" | egrep -o '[[0-9]]+\.[[0-9]]' | uniq | head -n1)
major=$(echo "$v_no" | egrep -o '^[[0-9]]+')
minor=$(echo "$v_no" | egrep -o '[[0-9]]$')
llvm_mentions=$(echo "$cxx_version" | grep -i -c -e 'apple llvm')
clang_mentions=$(echo "$cxx_version" | grep -c -e 'clang')
gxx_mentions=$(echo "$cxx_version" | grep -c -e '^g..')

check_openmp=false

if [[ "$sys_name" == "Darwin" ]]; then
    if [[[ "$llvm_mentions" -lt 1 && "$clang_mentions" -gt 0 ]]]; then
        if [[ "$major" -gt 3 ]]; then
            check_openmp=true
        elif [[[ "$major" == 3 && "$minor" -gt 7 ]]]; then
            check_openmp=true
        fi
    fi
elif [[ "$gxx_mentions" -gt 0 ]]; then
    if [[ "$major" -gt 5 ]]; then
        check_openmp=true
    elif [[[ "$major" == 5 && "$minor" -gt 3 ]]]; then
        check_openmp=true
    fi
fi

if [[ check_openmp ]]; then
    omp_notin_ldflags=$(${R_HOME}/bin/R CMD config --ldflags | grep -o fopenmp)
    if [[ -z "$omp_notin_ldflags" ]]; then
        AC_MSG_RESULT([not found, OpenMP turned off])
        openmp_flag='-DARMA_DONT_USE_OPENMP'
    else
        AC_MSG_RESULT([found, using OpenMP])
        openmp_flag='$(SHLIB_OPENMP_CXXFLAGS)'
    fi
else
    AC_MSG_RESULT([not found, OpenMP turned off])
    openmp_flag='-DARMA_DONT_USE_OPENMP'
fi

## Now set up Makevars
AC_SUBST([OPENMP_FLAG], ["${openmp_flag}"])
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

