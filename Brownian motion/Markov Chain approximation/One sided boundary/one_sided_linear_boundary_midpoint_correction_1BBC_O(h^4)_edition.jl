using Distributions

@doc """
	g(t, a = 1, b = 1) 

Linear boundary with intercept a and gradient b g(t) = a + b t
""" -> 
function g(t, a = 1, b = pi)
  	return a + b*t
end

@doc """
	exact_limit(T = 1 , a = 1, b = 1) 

Returns the exact probability that a Brownian motion crosses a linear boundary a + bt
""" -> 
function exact_limit(T = 1, a = 1, b = pi)
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


function pmatrix_end(n::Int, h, T = 1, lb = -3)
h2 = 3/n^2 
# h2 = 3/n^(9/4)
jrange = (g(T*(n-1)/n)-h/2):(-h):(lb) # moving from i to i+1
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
    if (g(T) - lb < c*h) | (x0 > g(0))
        return 1
    end
    if n == 1
    	prob = transpose(pmatrix0(n, c*h, T, x0, lb))
		for i = 1:(n-1)
			prob = prob*pmatrix(i, n, c*h, T, lb)
		end
		return 1 - (sum(prob))
    end
	prob = transpose(pmatrix0(n, c*h, T, x0, lb))
	for i = 1:(n-2)
		prob = prob*pmatrix(i, n, c*h, T, lb)
	end
		prob = prob*pmatrix_end(n, c*h, T, lb)
	return 1 - (sum(prob))
end

