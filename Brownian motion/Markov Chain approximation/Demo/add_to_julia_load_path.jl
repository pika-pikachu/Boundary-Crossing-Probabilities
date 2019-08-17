
# Adds to the load path
function load_module(PKG_NAME, PKG_VER)
	DEMO_LOCATION = "\\Users\\vince\\OneDrive\\Vincent's stuff\\University\\Research\\Code\\GitHub\\Boundary-Crossing-Probabilities\\Brownian motion\\Markov Chain approximation\\Demo"
	include(string(DEMO_LOCATION,"\\",PKG_VER ,"\\",PKG_NAME,".jl"))	
end

# loads the BCP algorithm for the final space modification
load_module("BM_BCP_BBC", "h4BBC")

# loads convergence plots
load_module("convergence_plot", "global")