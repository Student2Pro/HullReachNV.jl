# HullReachNV

[![Build Status](https://travis-ci.com/Student2Pro/HullReachNV.jl.svg?branch=master)](https://travis-ci.com/Student2Pro/HullReachNV.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/Student2Pro/HullReachNV.jl?svg=true)](https://ci.appveyor.com/project/Student2Pro/HullReachNV-jl)
[![Codecov](https://codecov.io/gh/Student2Pro/HullReachNV.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Student2Pro/HullReachNV.jl)

This package is based on https://github.com/sisl/NeuralVerification.jl

There are 4 methods in this package: MaxSens, HullReach, SGSV, SCH
HullReach is my improvment of MaxSens using hull-preserving
SGSV is improvment of MaxSens, published by the authors of MaxSens
SCH is my improvment of SGSV using hull-preserving

Hull-preserving is a property of certain extended set function. If an extended set function is hull-preserving, then for any simple region input set, the boundary of output set only depends on the boundary of input set.
