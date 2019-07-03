using Distributions

@doc """
	g(t, theta = 1) 

Returns the Daniel's boundary at time t. 
T is the terminal time of the boundary crossing 
theta affects the the Daniel's boundary at time 0
""" -> 
function g(t, theta = 1)
  	return 1
end


@doc """
	exact_limit(T = 1 , theta = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary with parameter theta
""" -> 
function exact_limit(T = 1, theta = 1)
	return 2*(1-cdf(Normal(0,1),1))
end


@doc """
	C(x, dt, h, lb)

Combines the transition probabilities of the lower tail
x: starting position
dt: time step size
h: space step size
""" -> 
function C(x, t0, t1, h, lb, mlb = -12)
range = lb:(-h):(-mlb)
l = length(range)
vec = zeros(l)
for j = 1:l
	vec[j] = transprob(x, range[j], t0, t1, h)
end
return sum(vec)
end

@doc """
	transprob(x, y, dt, h)

Returns the transition probability of Brownian motion
x: starting position
y: ending position
dt: time step size
h: space step size
""" -> 
function transprob(x, y, t0, t1, h, λ = 3)
dt = t1 - t0
	if y > x
		return λ*dt*exp(-(y-x)/0.15)/0.15*h  + (1 - λ*dt)*exp(-(y-x)^2/(2*dt))/sqrt(2*pi*dt)*h*(1 - exp(-2/(t1-t0)*(g(t1) - y)*(g(t0) - x)))
	else 
		return exp(-(y-x)^2/(2*dt))/sqrt(2*pi*dt)*h*(1 - exp(-2/(t1-t0)*(g(t1) - y)*(g(t0) - x)))
	end
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
function pmatrix0(n::Int, h, T = 1, x0 = 0, lb = -3)
range = (g(T/n)-h/2):(-h):(lb)
l = length(range)
lb = range[end]
vec = zeros(l)
	for j = 1:(l-1)
		vec[j] = transprob(x0, range[j], 0, T/n , h)
	end
vec[end] = C(x0, 0, T/n, h, lb) 
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
function pmatrix(i::Int, n::Int, h, T = 1, lb = -3)
jrange = (g(T*i/n)-h/2):(-h):(lb) # moving from i to i+1
krange = (g(T*(i+1)/n)-h/2):(-h):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = transprob(jrange[j], krange[k], T*i/n,T*(i+1)/n, h)
		end
		M[j, length(krange)] = C(jrange[j], T*i/n, T*(i+1)/n, h, lb)
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
function BCP(n::Int, h, T = 1, x0 = 0, lb = -3)
    if (g(T) - lb < h) | (x0 > g(0))
        return 1
    end
	prob = transpose(pmatrix0(n, h, T, x0, lb))
	for i = 1:(n-1)
		prob = prob*pmatrix(i, n, h, T, lb)
	end
	return 1-sum(prob)
end


using PyPlot
# using PyCall
# @pyimport matplotlib.patches as patch
using CPUTime


@doc """
	guideplot(N, s)

Returns a plot with 4 convergence lines n^{-1/2}, n^{-1}, n^{-2}, n^{-4}
N: Maximum number of boundary partitions
s: constant scaling
""" -> 
function guideplot(N, s = 0.05)
	# plt.xscale("log")
	# plt.yscale("log")
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	pvec = [0.5, 1,  2,  3]
	for i in 1:length(pvec)
		plot(1:N, 1 ./( (1:N).^(pvec[i]) )*s, ls = "--", color = "black")
		plt[:text](N + 0.3, 0.5*s/N^(pvec[i]) , join(["n^-", pvec[i]]), fontsize = 9)
	end	
	xticks(unique(vcat(1:10, 10:10:100, 100:100:1000, 1000:1000:10000)))
end


@doc """
	converge(n, N, p, T, x0, lb, γ)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions

converge(10,100,1,1,0,-3,6)
""" -> 
function converge(n, N, p = 1, T = 1, x0 = 0, lb = -3, γ = 1)
limit = exact_limit()

# cuts up the x-log axis uniformly to save computation
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) 
bcp_vec = zeros(length(n_mesh))
time_vec = zeros(length(n_mesh))
	for i in 1:length(n_mesh)
		CPUtic()
		bcp_vec[i] = abs(BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) - limit)
		time_vec[i] = CPUtoc()
	end
figure()
# guideplot(N, bcp_vec[1])
guideplot(N, N^(2*p)*bcp_vec[end])
plot(n_mesh, bcp_vec, marker="o")

# time elapsed plotting
	for i in 1:length(n_mesh)
		plt[:text](n_mesh[i] + 0.3, bcp_vec[i], join([time_vec[i],"s"]), fontsize = 9)
	end
xlabel(L"n")
ylabel(L"|P_n - P|")
title("Convergence Plot")
end



function converge_deriv(N, N0 = 1, p = 1, T = 1, x0 = 0, lb = -4, γ = 1)
n_mesh = N0:N
bcp_vec = zeros(length(n_mesh))
time_vec = zeros(length(n_mesh))
	for i in 1:length(n_mesh)
		CPUtic()
		bcp_vec[i] = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) 
		time_vec[i] = CPUtoc()
	end
figure(figsize=(10, 5))
	subplot(1,2,1)
		guideplot(N, bcp_vec[end]/(N^1))
		plot(n_mesh[2:end], abs.(bcp_vec[2:end] - bcp_vec[1:end-1]), marker="o", label = "BBC")
		xlabel(L"n")
		ylabel(L"|δP_n/δn|")
		title( join(["Convergence derivative δP_n/δn ","p=",p, ","," lb=",lb]))
		legend()
	subplot(1,2,2)
		plot(n_mesh,bcp_vec, marker="o", label = "BBC")
		xlabel(L"n")
		ylabel(L"P_n")
		title("BCP")
		legend()
end

