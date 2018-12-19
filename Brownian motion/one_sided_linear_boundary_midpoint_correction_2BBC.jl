using Distributions

@doc """
	g(s, T = 1 , a = 1, b = 1) 

T is the terminal time of the boundary crossing 
""" -> 
function g(s, T = 1, a = 1, b = 1)
	t = s.*T
  	return (a + b*t)/sqrt(T)
end

@doc """
	exact_limit(T = 1 , a = 1, b = 1) 

Returns the exact probability that a Brownian motion crosses a linear boundary
""" -> 
function exact_limit(T = 1, a = 1, b = 1)
	if T == 0 
	  return 0
	end
	b1 = cdf(Normal(0 , 1), b*sqrt(T) + a/sqrt(T))
	b2 = exp(-2*a*b)*cdf(Normal(0, 1), b*sqrt(T) - a/sqrt(T)) 
	return 1 - b1 + b2
end

@doc """
	J(a1, b1, a2, b2, t1, t2, x)

Returns the exact probability that a Brownian bridge crosses two piecewise linear boundary:
a1 + b1 t, 0 <= t <= t1
a2 + b2 t, t1 <= t <= t2
""" -> 
function J(a1, b1, a2, b2, t1, t2, x)
h = (a1 + b1*t1)/t1 - x/t2
T = sqrt(t2*t1/(t2-t1))
a2d = b2 + (a2 - x)/t2
a1d = a1*(b1 + (a1 - x)/t2)
J = 1 - cdf(Normal(),h*T) + exp(-2*a2*a2d)*
    cdf(Normal(), (h - 2*a2d)*T ) +
    exp(-2*a1d)* cdf(Normal(), h*T - 2*a1/T) -
    exp(-2*a1d + (4*a1- 2*a2)*a2d )*
    cdf(Normal(), (h - 2*a2d)*T - 2*a1/T )
return J
end

@doc """
	bbb(x0, x1, t0, t2, T) 

Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
a two piecewise linear boundary approximation of g(t):
g(t), 0  <= t <= t1
g(t), t1 <= t <= t2
""" -> 
function bbb(x0, x1, t0, t2, T)
        t1 = (t0 + t2)/2 #half point between times
        b1 = (g(t1,T) - g(t0,T))/(t1 - t0) 
        a1 = g(t0,T) - x0
        b2 = (g(t2,T) - g(t1,T))/(t2 - t1)
        a2 = g(t2,T) - b2*(t2 - t0) - x0
        return 1 - J(a1, b1, a2, b2, t1 - t0, t2 - t0, x1 - x0)
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
	pmatrix0(n, h, T, lb)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n
n: number of time partitions
h: space step size
T: Terminal time
lb: Lower bound for truncation
""" -> 
function pmatrix0(n::Int, h, T, lb = -3)
range = (g(1/n,T)-h/2):(-h):(lb)
l = length(range)
lb = range[end]
vec = zeros(l)
	for j = 1:(l-1)
		vec[j] = bbb(0, range[j], 0, 1/n, T)*transprob(0, range[j], 1/n, h)
	end
vec[end] = bbb(0, lb, 0, 1/n, T)*C(0, 1/n, h, lb) 
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
function pmatrix(i::Int, n::Int, h, T, lb = -3)
jrange = (g(i/n,T)-h/2):(-h):(lb) # moving from i to i+1
krange = (g((i+1)/n,T)-h/2):(-h):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
				M[j, k] = bbb(jrange[j], krange[k], i/n, (i+1)/n, T)*transprob(jrange[j], krange[k], 1/n, h)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, i/n, (i+1)/n, T)*C(jrange[j], 1/n, h, lb)
	end
M[length(jrange), length(krange)] = 1
return M
end


@doc """
	BCP(n::Int, h, T, lb)

Returns the approximated boundary crossing probability
n: number of time partitions
h: space step size
T: Terminal time
lb: Lower bound for truncation
""" -> 
function BCP(n::Int, h, T = 1, lb = -3)
    if g(T, T) - lb < h
        return 1
    end
	prob = transpose(pmatrix0(n, h, T, lb))
	for i = 1:(n-1)
		prob = prob*pmatrix(i, n, h, T, lb)
	end
	return 1 - (sum(prob))
end

