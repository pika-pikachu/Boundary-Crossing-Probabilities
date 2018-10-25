# File descriptions

One_sided_Daniels_boundary_midpoint_correction.jl
- Returns the approximated probability a Brownian motion trajectory crosses Daniel's boundary
- Uses a two-piece Brownian Bridge Correction (BBC) 
- Uses a midpoint correction for the state space
- Uses a boundary dependent state space

kolmogorov_smirnov_novikov.jl
- Returns the approximated probability a Brownian motion trajectory crosses either boundary an upper boundary 3sqrt(t) or lower boundary -3sqrt(t)
- Uses a one-piece Brownian Bridge Correction
- Uses a midpoint correction for the state space 
