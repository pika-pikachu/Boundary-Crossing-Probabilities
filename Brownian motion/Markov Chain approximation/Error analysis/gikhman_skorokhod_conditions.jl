###############################################################
# Markov chain approximation of BM for boundary crossing probabilities
#
#
# Bugs:
# - Grids are relative to x0 = 0, convergence is not guaranteed otherwise.
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
	constC0(x, n, h, T)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n

n: number of time partitions
h: space step size
T: Terminal time
x0: Starting position
lb: Lower bound for truncation
""" -> 
function constC0(n, h, T=1)
dt = 1/n
range1 = 0:h:5
range2 = (-h):(-h):-5
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
for i in 1:length(range1) # 
	vec1[i] = exp(-range1[i]^2/(2*dt))/sqrt(2*pi*dt)*h
end
for i in 1:length(range2)
	vec2[i] = exp(-range2[i]^2/(2*dt))/sqrt(2*pi*dt)*h
end
C_h = sum(vec1) + sum(vec2)
	return C_h
end


@doc """
	constCi(i, x, n, h, T)

Returns the normalising constant for the transition of BM
from time i/n to time (i+1)/n

n: number of time partitions
h: space step size
T: Terminal time
x0: Starting position
lb: Lower bound for truncation
""" -> 
function constCi(i, x, n, h, T=1)
dt = 1/n
range1 = 0:h:5
range2 = (-h):(-h):-5
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
	for i in 1:length(range1)
		vec1[i] = exp(-(range1[i]-x)^2/(2*dt))/sqrt(2*pi*dt)*h
	end
	for i in 1:length(range2)
		vec2[i] = exp(-(range2[i]-x)^2/(2*dt))/sqrt(2*pi*dt)*h
	end
C_h = sum(vec1) + sum(vec2)
	return C_h
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
hk = g(T/n)/floor(1/h) # boundary dependent spacing to ensure that the mean transition is zero
krange = g(T/n):(-hk):(lb) # boundary dependent mesh
l = length(krange)
lb = krange[end] # boundary dependent lower boundary
p_vec = zeros(l) 
c0 = constC0(n, hk)
for k = 1:(l-1)
	p_vec[k] = bbb(x0, krange[k], 0, T/n)*transprob(x0, krange[k], T/n, hk)/c0
end
p_vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, hk, lb)/c0
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
hj = g(T*i/n)/floor(1/h) # boundary dependent spacing to ensure that the mean transition is zero
jrange = g(T*i/n):(-hj):(lb) # moving from i to i+1
hk = g(T*(i+1)/n)/floor(1/h)
krange = g(T*(i+1)/n):(-hk):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		cj = constCi(i,jrange[j],n,hk,T) # scaling constant is depedendent on the starting point
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*i/n, T*(i+1)/n)*transprob(jrange[j], krange[k], T/n, hk)/cj
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n)*C(jrange[j], T/n, hk, lb)/constCi(i,jrange[j],n,hk,T)
	end
M[length(jrange), length(krange)] = 1
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
hj = g(T*(n-1)/n)/floor(1/h)
jrange = g(T*(n-1)/n):(-hj):(lb) 
h2 = 1/n
krange = (g(T)-h2/2):(-h2):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*(n-1)/n, T)*transprob(jrange[j], krange[k], T/n, h2)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*(n-1)/n, T)*C(jrange[j], T/n, h2, lb)
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
function BCP(n::Int, h, T = 1, x0 = 0, lb = -3, c = 1)
	h = 1/sqrt(n)
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



