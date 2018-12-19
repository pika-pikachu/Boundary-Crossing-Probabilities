# File descriptions

convergence_plot.jl
- Plots the convergence behaviour of the Markov chain approximation of a Brownian boundary crossing probability

integral_equation_loader_deely.jl
- Using the method of integral equations, returns the approximated probability a Brownian motion trajectory crosses Daniel's boundary

kolmogorov_smirnov_novikov.jl
- Returns the approximated probability a Brownian motion trajectory crosses either boundary an upper boundary 3sqrt(t) or lower boundary -3sqrt(t)
- Uses a one-piece Brownian Bridge Correction
- Uses a midpoint correction for the state space 

one_sided_daniels_boundary_midpoint_correction_1BBC.jl
- Returns the Markov chain approximated probability a Brownian motion trajectory crosses Daniel's boundary
- Uses a two-piece Brownian Bridge Correction (BBC) 
- Uses a midpoint correction for the state space
- Uses a boundary dependent state space
- Contains exact value for comparison

one_sided_daniels_boundary_midpoint_correction_2BBC.jl
- Uses a two-piece Brownian Bridge Correction (BBC) 

one_sided_linear_boundary_midpoint_correction_1BBC.jl
- Returns the Markov chain approximated probability a Brownian motion trajectory crosses a linear boundary

one_sided_linear_boundary_midpoint_correction_2BBC.jl
- Uses a two-piece Brownian Bridge Correction (BBC) 

two_sided_boundary_midpoint_correction.jl
- Returns the Markov chain approximated probability a Brownian motion trajectory crosses a two sided boundary
- Uses a one-piece Brownian Bridge Correction (BBC) 
- Uses a midpoint correction for the state space
- Contains exact value for comparison