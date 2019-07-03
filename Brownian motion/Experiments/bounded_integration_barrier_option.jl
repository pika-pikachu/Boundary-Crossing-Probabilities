using Distributions

@doc """
	g(t, theta = 1) 

Returns the Daniel's boundary at time t. 
T is the terminal time of the boundary crossing 
theta affects the the Daniel's boundary at time 0
""" -> 
function g(t, theta = 1)
	if t == 0
	  return theta/2
	end
	a1 = a2 = 1/2
	a = 1
  	return theta/2 - t/theta*log(0.5*a1/a + sqrt( 1/4*(a1/a)^2 + (a2/a)*exp(-theta^2/t) ) )
end


@doc """
	exact_limit(T = 1 , theta = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary with parameter theta
""" -> 
function exact_limit(T = 1, theta = 1, lower = 0)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)),g(T)/sqrt(T)) - cdf(Normal(0 ,sqrt(T)), lower/sqrt(T))
	b2 = a1*(cdf(Normal(theta,sqrt(T)),g(T)/sqrt(T)) - cdf(Normal(theta,sqrt(T)), lower/sqrt(T)) )
	b3 = a2*(cdf(Normal(2*theta,sqrt(T)),g(T)/sqrt(T)) - cdf(Normal(2*theta,sqrt(T)), lower/sqrt(T)) )  
	return 1 - (b1 - (b2 + b3)/a)
end


# function exact_limit(T = 1, theta = 1)
# 	if T == 0 
# 	  return 0
# 	end
# 	a1 = a2 = 1/2
# 	a = 1
# 	b1 = cdf(Normal(0 ,sqrt(T)),g(T)/sqrt(T)) 
# 	b2 = a1*(cdf(Normal(theta,sqrt(T)),g(T)/sqrt(T)))
# 	b3 = a2*(cdf(Normal(2*theta,sqrt(T)),g(T)/sqrt(T)) )  
# 	return 1 - (b1 - (b2 + b3)/a)
# end


@doc """
	bbb(x0, x1, t0, t1) 

Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
a two piecewise linear boundary approximation of g(t):
g(t), t0  <= t <= t1
""" -> 
function bbb(x0, x1, t0, t1)
    return 1 - exp(-2/(t1-t0)*(g(t1) - x1)*(g(t0) - x0))
end

@doc """
	C(x, dt, h, lb)

Combines the transition probabilities of the lower tail
x: starting position
dt: time step size
h: space step size
""" -> 
function C(x, dt, h, lb)
range = lb:(-h):(-12)
l = length(range)
vec = zeros(l)
for j = 1:l
	vec[j] = transprob(x, range[j], dt, h)
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
function transprob(x, y, dt, h)
	return exp(-(y-x)^2/(2*dt))/sqrt(2*pi*dt)*h
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
		vec[j] = bbb(x0, range[j], 0, T/n)*transprob(x0, range[j], T/n, h)
	end
vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, h, lb) 
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
			M[j, k] = bbb(jrange[j], krange[k], T*i/n, T*(i+1)/n)*transprob(jrange[j], krange[k], T/n, h)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n)*C(jrange[j], T/n, h, lb)
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
	return prob
end

# n = 10
# h = 1/n
# x = BCP(n, h)
# lb = -3
# lower = 0
# plot((g(1)-h/2):(-h):(lb),x)

# grid = (g(1)-h/2):(-h):(lb)
# sum(x[grid.> 0])

# 1 - exact_limit(1,1,lower)



using PyPlot
# using PyCall
# @pyimport matplotlib.patches as patch

@doc """
	guideplot(N, s)

Returns a plot with 4 convergence lines n^{-1/2}, n^{-1}, n^{-2}, n^{-4}
N: Maximum number of boundary partitions
s: constant scaling
""" -> 
function guideplot(N, s = 0.05)
	plt[:xscale]("log")
	plt[:yscale]("log")
	grid("on", lw = 0.5)
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s, label = L"1/n", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^1.5)*s, label = L"1/n^1.5", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s, label = L"1/n^2", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s, label = L"1/n^4", ls=  "--", color = "black")
	# plot(1:N, 1 ./((1:N).^6)*s, label = L"1/n^6", ls=  "--", color = "black")
	xticks(unique(vcat(1:10, 10:10:100)))
end


@doc """
	converge(n, N, p, T, x0, lb)

Returns a convergence plot of the MC approximation towards the true solution
n: number of equally spaced points
N: maximum number of boundary partitions

converge(10,100,1,1,0,-3,6)
""" -> 
function converge(n, N, p = 1, T = 1, x0 = 0, lb = -3, γ = 1, lb2 = -4, lb_trans = 10^6)
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) # cuts up the x -l og axis uniformly to save computation
bcp_vec = zeros(length(n_mesh))
bcp_vec2 = zeros(length(n_mesh))
lower = 0
for i in 1:length(n_mesh)
	if i > lb_trans
		lb = lb2
	end
	x = BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb)
	h = 1/n_mesh[i]
	grid = (g(1)-h/2):(-h):(lb)
	index = grid.> lower
	grid2 = grid[index]
	lower2 = grid2[end] - h/2
	limit = 1 - exact_limit(1,1,lower2)
	limit2 = 1 - exact_limit(1,1,lower)
	approx = sum(x[index])
	bcp_vec[i] = abs(approx - limit)
	bcp_vec2[i] = abs(approx - limit2)
end
figure()
guideplot(N, bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
plot(n_mesh, bcp_vec2, marker="*")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
end


