###############################################################
# Markov chain approximation of BM for boundary crossing probabilities
#
#
# Bugs:
# - Grids are relative to x0 = 0, convergence is not guaranteed otherwise.
# - It is unclear if the diffusion coefficient converges
###############################################################

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
function exact_limit(T = 1, theta = 1)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)),g(T)/sqrt(T)) 
	b2 = a1*cdf(Normal(theta,sqrt(T)),g(T)/sqrt(T)) 
	b3 = a2*cdf(Normal(2*theta,sqrt(T)),g(T)/sqrt(T)) 
	return 1 - (b1 - (b2 + b3)/a)
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
	C(x, dt, h, lb)

Combines the transition probabilities of the lower tail
x: starting position
dt: time step size
h: space step size
""" -> 
function C(x, dt, h, lb, inf = 5)
lower_range = lb:(-h):(-inf)
lower_p_vec = zeros(length(lower_range))
	for j = 1:length(lower_range)
		lower_p_vec[j] = transprob(x, lower_range[j], dt, h)
	end
return sum(lower_p_vec)
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
	constC(x, n, h, T)

Returns the normalising constant for the transition of BM
from for starting position x 

x: starting position
n: number of time partitions
h: space step size
T: Terminal time
""" -> 
function constC(x, n, h, T=1, inf = 5)
range1 = 0:h:inf
range2 = (-h):(-h):(-inf-h)
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
	for i in 1:length(range1)
		vec1[i] = exp(-n*(range1[i]-x)^2/2)*sqrt(n/2/pi)*h
		vec2[i] = exp(-n*(range2[i]-x)^2/2)*sqrt(n/2/pi)*h
	end
return sum(vec1) + sum(vec2)
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
h_k = g(T/n)/floor(1/h) # boundary dependent spacing to ensure that the mean transition is zero
k_range = g(T/n):(-h_k):(lb) # boundary dependent mesh
l = length(k_range)
lb = k_range[end] # boundary dependent lower boundary
p_vec = zeros(l) 
c0 = constC(0, n, h_k, T)
	for k = 1:(l-1)
		p_vec[k] = bbb(x0, k_range[k], 0, T/n)*transprob(x0, k_range[k], T/n, h_k)/c0
	end
p_vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, h_k, lb)/c0
return p_vec
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
h_j = g(T*i/n)/floor(1/h) # boundary dependent spacing to ensure that the mean transition is zero
j_range = g(T*i/n):(-h_j):(lb) # moving from i to i+1
h_k = g(T*(i+1)/n)/floor(1/h)
k_range = g(T*(i+1)/n):(-h_k):(lb)
lb = k_range[end]
M = zeros(length(j_range),length(k_range))
	for j = 1:(length(j_range)-1)
		cj = constC(j_range[j], n, h_k, T) # scaling constant is depedendent on the starting point
		for k = 1:(length(k_range)-1)
			M[j, k] = bbb(j_range[j], k_range[k], T*i/n, T*(i+1)/n)*transprob(j_range[j], k_range[k], T/n, h_k)/cj
		end
		M[j, length(k_range)] = bbb(j_range[j], lb, T*i/n, T*(i+1)/n)*C(j_range[j], T/n, h_k, lb)/constC(j_range[j], n, h_k, T)
	end
M[length(j_range), length(k_range)] = 1
return M
end


@doc """
	pmatrix_end(i, n, h, T, lb)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion
from time (n-1)/n to time 1
i: ith time partition 
n: number of time partitions
h: space step size
T: Terminal time
lb: Lower bound for truncation
""" -> 
function pmatrix_end(n::Int, h, T = 1, lb = -3)
h_j = g(T*(n-1)/n)/floor(1/h)
j_range = g(T*(n-1)/n):(-h_j):(lb) 
h_n = 1/n
k_range = (g(T)-h_n/2):(-h_n):(lb)
lb = k_range[end]
M = zeros(length(j_range),length(k_range))
	for j = 1:(length(j_range)-1)
		for k = 1:(length(k_range)-1)
			M[j, k] = bbb(j_range[j], k_range[k], T*(n-1)/n, T)*transprob(j_range[j], k_range[k], T/n, h_n)
		end
		M[j, length(k_range)] = bbb(j_range[j], lb, T*(n-1)/n, T)*C(j_range[j], T/n, h_n, lb)
	end
M[length(j_range), length(k_range)] = 1
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
function BCP(n::Int, h = 1, T = 1, x0 = 0, lb = -3, c = 1)
	h = 1/sqrt(n)
    if (g(T) - lb < c*h) | (x0 > g(0))
        return 1
    end
    if n == 1
    	prob = transpose(pmatrix0(n, c*h, T, x0, lb))
		return 1 - (sum(prob))
    else
		prob = transpose(pmatrix0(n, c*h, T, x0, lb))
		for i = 1:(n-2)
			prob = prob*pmatrix(i, n, c*h, T, lb)
		end
		prob = prob*pmatrix_end(n, c*h, T, lb)
	end
	return 1 - sum(prob)
end



function RichardsonExtrap(N, lb = -3, p = 2, q = 2, l = 1, x0 = 0, T = 1)
M = zeros(N, N)
	function fn(n)
		return 1/n^l
	end
	function A(n)
		return BCP(n, fn(n), T, x0, lb)
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

