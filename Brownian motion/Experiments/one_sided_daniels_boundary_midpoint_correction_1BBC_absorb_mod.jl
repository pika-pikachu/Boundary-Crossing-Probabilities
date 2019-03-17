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
	pL(x, dt, g1, g2, L)

Closed form expression for P(tau > T, W_T < x).
x: starting position
dt: time step size
g1: linear boundary at t1
g2: linear boundary at t2
L: Lower absorbing boundary
""" -> 
function pL(x, dt, g1, g2, L)
	return cdf(Normal(),(L-x)/sqrt(dt)) - exp(2/dt*(g1-g2)*(g1-x))*ccdf(Normal(),(2*g1 - L- x)/sqrt(dt))
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
vec[end] = pL(0, T/n, g(0), g(T/n), lb - h/2)
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
		M[j, length(krange)] = pL(jrange[j], T/n, g(T*i/n), g(T*(i+1)/n), lb - h/2)
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
	return 1 - (sum(prob))
end


