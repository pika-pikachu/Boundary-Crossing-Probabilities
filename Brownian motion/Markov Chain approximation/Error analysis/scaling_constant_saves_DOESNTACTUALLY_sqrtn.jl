using Distributions

# @doc """
# 	g(t, theta = 1) 

# Returns the Daniel's boundary at time t. 
# T is the terminal time of the boundary crossing 
# theta affects the the Daniel's boundary at time 0
# """ -> 
# function g(t, theta = 1)
# 	if t == 0
# 	  return theta/2
# 	end
# 	a1 = a2 = 1/2
# 	a = 1
#   	return theta/2 - t/theta*log(0.5*a1/a + sqrt( 1/4*(a1/a)^2 + (a2/a)*exp(-theta^2/t) ) )
# end


# @doc """
# 	exact_limit(T = 1 , theta = 1) 

# Returns the exact probability that a Brownian motion crosses Daniel's boundary with parameter theta
# """ -> 
# function exact_limit(T = 1, theta = 1)
# 	if T == 0 
# 	  return 0
# 	end
# 	a1 = a2 = 1/2
# 	a = 1
# 	b1 = cdf(Normal(0 ,sqrt(T)),g(T)/sqrt(T)) 
# 	b2 = a1*cdf(Normal(theta,sqrt(T)),g(T)/sqrt(T)) 
# 	b3 = a2*cdf(Normal(2*theta,sqrt(T)),g(T)/sqrt(T)) 
# 	return 1 - (b1 - (b2 + b3)/a)
# end

@doc """
	g(t, a = 1, b = 1) 

Linear boundary with intercept a and gradient b g(t) = a + b t
""" -> 
function g(t, a = 3, b = pi/2)
  	return a + b*t
end

@doc """
	exact_limit(T = 1 , a = 1, b = 1) 

Returns the exact probability that a Brownian motion crosses a linear boundary a + bt
""" -> 
function exact_limit(T = 1, a = 3, b = pi/2)
	if T == 0 
	  return 0
	end
	b1 = cdf(Normal(0 , 1), b*sqrt(T) + a/sqrt(T))
	b2 = exp(-2*a*b)*cdf(Normal(0, 1), b*sqrt(T) - a/sqrt(T)) 
	return 1 - b1 + b2
end



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
	cscale(i, n, h)

Scales the transition probabilities from time step i to time step i+1, in order to make the transition probabilities add up to one
n: number of time partitions
h: space step size
""" -> 
function cscale(i, n, h)
eps = g((i+1)/n) - g(i/n)  # Relative shift of the grid from i to i+1
range1 = eps:h:12
range2 = (eps-h):(-h):(-12)
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
for j = 1:length(range1)
	vec1[j] = exp(-n*range1[j]^2/2)
end
for j = 1:length(range2)
	vec2[j] = exp(-n*range2[j]^2/2)
end
return sqrt(n/(2*pi))*(sum(vec1)+sum(vec2))*h
end

@doc """
	C(x, dt, h, lb)

Combines the transition probabilities of the lower tail
x: starting position
dt: time step size
h: space step size
""" -> 
function C(x, dt, h, lb)
range = lb:(-h):(-6)
l = length(range)
vec = zeros(l)
for j = 1:l
	vec[j] = transprob(x, range[j], dt, h)
end
return sum(vec)
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
# vec[end] = C(x0, T/n, h, lb) 
vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, h, lb) 
return vec/cscale(0,n,h)
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
		# M[j, length(krange)] = C(jrange[j], T/n, h, lb)
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n)*C(jrange[j], T/n, h, lb)
	end
M = M/cscale(i,n,h)
M[length(jrange), length(krange)] = 1
return M
end


function pmatrix_end(n::Int, h, T = 1, lb = -3)
h2 = 1/n
jrange = (g(T*(n-1)/n)-h/2):(-h):(lb) 
krange = (g(T)-h2/2):(-h2):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*(n-1)/n, T)*transprob(jrange[j], krange[k], T/n, h2)
		end
		# M[j, length(krange)] = C(jrange[j], T/n, h2, lb)
		M[j, length(krange)] = bbb(jrange[j], lb, T*(n-1)/n, T)*C(jrange[j], T/n, h2, lb)
		# M[j, length(krange)] = bbb(jrange[j], lb, T*(n-1)/n, T)*cdf(Normal(jrange[j], sqrt(T/n)), lb)
	end
M = M/cscale(n-1,n, h2)
M[length(jrange), length(krange)] = 1
return M
end

@doc """
	BCP(n::Int, h, T, x0, lb)

Returns the approximated boundary crossing probability
n: number of time partitions
h: space step size (set h(n) = n^-0.52 for a good time)
T: Terminal time
x0: Initial position of Wiener process
lb: Lower bound for truncation
""" -> 
function BCP(n::Int, h, T = 1, x0 = 0, lb = -3, c = 1)
    if (g(T) - lb < c*h) | (x0 > g(0))
        return 1
    end
    if n == 1
    	prob = transpose(pmatrix0(n, c*h, T, x0, lb))
		for i = 1:(n-1)
			prob = prob*pmatrix(i, n, c*h, T, lb)
		end
		return 1 - (sum(prob))
    else
		prob = transpose(pmatrix0(n, c*h, T, x0, lb))
		for i = 1:(n-2)
			prob = prob*pmatrix(i, n, c*h, T, lb)
		end
		prob = prob*pmatrix_end(n, c*h, T, lb)
	end
	return 1 - (sum(prob))
end




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
	plt.xscale("log")
	plt.yscale("log")
	grid("on", lw = 0.5)
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s, label = L"1/n", ls = "--", color = "black")
	# plot(1:N, 1 ./((1:N).^1.5)*s, label = L"1/n^1.5", ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s, label = L"1/n^2", ls = "--", color = "black")
	# plot(1:N, 1 ./((1:N).^3.5)*s, label = L"1/n^3.5", ls=  "--", color = "black")
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
function converge(n, N, p = 1/2, T = 1, x0 = 0, lb = -3, γ = 1, lb2 = -4, lb_trans = 10^6)
limit = exact_limit()
n_mesh = unique(floor.(Int,exp.( log(N)*( 0:(1/(n-1)):1 ) ) ) ) # cuts up the x -l og axis uniformly to save computation
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	if i > lb_trans
		lb = lb2
	end
	bcp_vec[i] = abs(BCP(n_mesh[i], γ/n_mesh[i]^p, T, x0, lb) - limit)
end
figure()
guideplot(N, bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
end




function RichardsonExtrap(N, lb = -3, p = 2, q = 2, x0 = 0, T = 1)
M = zeros(N, N)
	function A(n)
		return BCP(n, 1/sqrt(n), T, x0, lb)
	end
M[1, 1] = A(2^0)
for j = 2:N
	M[j,1] = A(q^(j-1))
	for k = 2:j
		M[j, k] = (q^((k-1)*p)*M[j, k-1] - M[j-1, k-1])/(q^((k-1)*p)-1)
	end
end
return M
end


function convergeRE(N)
limit = exact_limit()
n_mesh = 2 .^(0:(N-1))
bcp_mat = RichardsonExtrap(N)
bcp_RE_vec = zeros(N)
bcp_vec = zeros(length(n_mesh))
for i in 1:length(n_mesh)
	bcp_RE_vec[i] = abs(bcp_mat[i,i] - limit)
	bcp_vec[i] = abs(BCP(n_mesh[i], 1/sqrt(n_mesh[i])) - limit)
end
figure()
guideplot(2^(N-1), bcp_vec[1])
plot(n_mesh, bcp_vec, marker="o")
plot(n_mesh, bcp_RE_vec, marker="x")
xlabel("n")
ylabel(L"|P_n - P|")
title(L"Error, $|P - P_n|$")
end


