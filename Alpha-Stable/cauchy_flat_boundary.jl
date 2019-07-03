using Distributions

@doc """
	g(t, theta = 1) 

Returns the Daniel's boundary at time t. 
T is the terminal time of the boundary crossing 
theta affects the the Daniel's boundary at time 0
""" -> 
function g(t, theta = 1)
  	# return 1
  	# return 1.5
  	return 2
  	# return sqrt(t+1)
end


@doc """
	exact_limit(T = 1 , theta = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary with parameter theta
""" -> 
function exact_limit(T = 1, theta = 1)
# 1
	# return 0.329307 
# 1.5
	# return 0.233903	
# 2
	return 0.178164
end

@doc """
	bbb(x0, x1, t0, t1) 

Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
a two piecewise linear boundary approximation of g(t):
g(t), t0  <= t <= t1
""" -> 
function bbb(x0, x1, t0, t1, corr = 1)
	if corr == 1
    	return 1 - exp(-2/(t1-t0)*(g(t1) - x1)*(g(t0) - x0))
    else 
     	return 1
    end
end

@doc """
	C(x, dt, h, lb)

Combines the transition probabilities of the lower tail
x: starting position
dt: time step size
h: space step size
""" -> 
function C(x, dt, h, lb)
	return atan( (lb -h/2-x)/dt )/pi + 1/2
end

@doc """
	transprob(x, y, dt, h)

Returns the transition probability of Brownian motion
x: starting position
y: ending position
dt: time step size
h: space step size
""" -> 
function transprob(x, y, dt, h)
	return h*dt/pi/( dt^2 + (y-x)^2)
end

@doc """
	pmatrix0(n, h, T, x0, lb)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n
n: number of time partitions
h: space step size
T: Terminal time
x0: Starting position
lb: Lower bound for truncation
""" -> 
function pmatrix0(n::Int, h, T = 1, x0 = 0, lb = -3, corr = 1)
range = (g(T/n)-h/2):(-h):(lb)
l = length(range)
lb = range[end]
vec = zeros(l)
	for j = 1:(l-1)
		vec[j] = bbb(x0, range[j], 0, T/n, corr)*transprob(x0, range[j], T/n, h)
	end
vec[end] = bbb(x0, lb, 0, T/n, corr)*C(x0, T/n, h, lb) 
return vec
end

@doc """
	pmatrix(i, n, h, T, lb)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion
from time i/n to time (i+1)/n
i: ith time partition 
n: number of time partitions
h: space step size
T: Terminal time
lb: Lower bound for truncation
""" -> 
function pmatrix(i::Int, n::Int, h, T = 1, lb = -3, corr = 1)
jrange = (g(T*i/n)-h/2):(-h):(lb) # moving from i to i+1
krange = (g(T*(i+1)/n)-h/2):(-h):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*i/n, T*(i+1)/n, corr)*transprob(jrange[j], krange[k], T/n, h)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n, corr)*C(jrange[j], T/n, h, lb)
	end
M[length(jrange), length(krange)] = 1
return M
end


@doc """
	BCP(n::Int, h, T, x0, lb)

Returns the approximated boundary crossing probability
n: number of time partitions
h: space step size
T: Terminal time
x0: Initial position of Wiener process
lb: Lower bound for truncation
""" -> 
function BCP(n::Int, h, T = 1, x0 = 0, lb = -3, corr = 1)
    if (g(T) - lb < h) | (x0 > g(0))
        return 1
    end
	prob = transpose(pmatrix0(n, h, T, x0, lb, corr))
	for i = 1:(n-1)
		prob = prob*pmatrix(i, n, h, T, lb, corr)
	end
	return 1 - sum(prob)
end


# Checking tail probabilitiies
# n = 10
# h = 1/n^1.1
# lb = -15
# x = (g(1)-h/2):(-h):lb
# y = BCP(n,h,1,0,lb)'
# plot(x,y)


using PyPlot
using CPUTime


@doc """
	guideplot(N, s)

Returns a plot with 4 convergence lines n^{-1/2}, n^{-1}, n^{-2}, n^{-4}
N: Maximum number of boundary partitions
s: constant scaling
""" -> 
function guideplot(N, s = 0.05, offset = 0)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	pvec = [0.5, 0.75, 1, 1.5, 1.75]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.3, s/N^(pvec[i]) , join(["n^-", pvec[i] + offset]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end


function converge(n, N, p = 1.2, T = 1, x0 = 0, lb = -11, γ = 1)
limit = exact_limit()
# cuts up the x-log axis uniformly to save computation
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) 
bcp_vec = zeros(length(n_mesh))
bcp_nbb_vec = zeros(length(n_mesh))
bcp_avg_vec = zeros(length(n_mesh))
time_vec = zeros(length(n_mesh))
	for i in 1:length(n_mesh)
		CPUtic()
		bcp_vec[i] = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) 
		bcp_nbb_vec[i] = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb, 0) 
		bcp_avg_vec[i] = (bcp_vec[i] + bcp_nbb_vec[i])/2 
		time_vec[i] = CPUtoc()
	end
figure(figsize=(10, 5))
	subplot(1,2,1)
		guideplot(N, N^(0.75)*abs(bcp_nbb_vec[end]-limit))
		plot(n_mesh, abs.(bcp_vec .- limit), marker="o", label = "w/ BB")
		plot(n_mesh, abs.(bcp_nbb_vec .- limit), marker="o", label = "no BB")
		plot(n_mesh, abs.(bcp_avg_vec .- limit), marker="o", label = "avg")
	# time elapsed plotting
		for i in 1:length(n_mesh)
			plt[:text](n_mesh[i] + 0.3, abs(bcp_vec[i] - limit), join([time_vec[i],"s"]), fontsize = 9)
		end
		xlabel(L"n")
		ylabel(L"|P_n - P|")
		title( join(["Error convergence ","p=",p, ","," lb=",lb]))
		legend()
	subplot(1,2,2)
		plot(n_mesh, bcp_vec, marker="o", label = "bbc")
		plot(n_mesh, bcp_nbb_vec, marker="o", label = "no bbc")
		plot(n_mesh, bcp_avg_vec, marker="o", label = "average")
		plot(n_mesh, zeros(length(n_mesh)) .+ limit, color = "black", label= "exact")
		xlabel(L"n")
		ylabel(L"P_n")
		title("Convergence of P_n")
		legend()
end




# used to analyse non flat boundaries

function converge_deriv(N, N0 = 1, p = 1.2, T = 1, x0 = 0, lb = -11, γ = 1)
n_mesh = N0:N
bcp_vec = zeros(length(n_mesh))
bcp_nbb_vec = zeros(length(n_mesh))
bcp_avg_vec = zeros(length(n_mesh))
time_vec = zeros(length(n_mesh))
	for i in 1:length(n_mesh)
		CPUtic()
		bcp_vec[i] = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) 
		bcp_nbb_vec[i] = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb, 0) 
		bcp_avg_vec[i] = (bcp_vec[i] + bcp_nbb_vec[i])/2
		time_vec[i] = CPUtoc()
	end
figure(figsize=(10, 5))
	subplot(1,2,1)
		guideplot(N, 0.05, 0) # offset of -1
		plot(n_mesh[2:end], abs.(bcp_vec[2:end] - bcp_vec[1:end-1]), marker="o", label = "BBC")
		plot(n_mesh[2:end], abs.(bcp_nbb_vec[2:end] - bcp_nbb_vec[1:end-1]), marker="o", label = "No BBC")
		plot(n_mesh[2:end], abs.(bcp_avg_vec[2:end] - bcp_avg_vec[1:end-1]), marker="o", label = "avg")
		xlabel(L"n")
		ylabel(L"|δP_n/δn|")
		title( join(["Convergence derivative δP_n/δn ","p=",p, ","," lb=",lb]))
		legend()
	subplot(1,2,2)
		plot(n_mesh,bcp_vec, marker="o", label = "BBC")
		plot(n_mesh,bcp_nbb_vec, marker="o", label = "No BBC")
		plot(n_mesh,bcp_avg_vec, marker="o", label = "avg")
		xlabel(L"n")
		ylabel(L"P_n")
		title("BCP")
		legend()
end


