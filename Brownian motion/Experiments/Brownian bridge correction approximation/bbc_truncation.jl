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
n = 1/(t1-t0)
d1 = g(t1) - x1
d0 = g(t0) - x0
	if d1*d0 < 1/sqrt(n)
		return 1- exp(-2/(t1-t0)*d1*d0)
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
function C(x, dt, h, lb, p_trunc, c_trunc)
tail_mesh = lb:(-h):(-12)
l = length(tail_mesh)
tail_vec = zeros(l)
for j = 1:l
	tail_vec[j] = transprob(x, tail_mesh[j], dt, h, p_trunc, c_trunc)
end
return sum(tail_vec)
end

@doc """
	transprob(x, y, dt, h)

Returns the transition probability of Brownian motion
x: starting position
y: ending position
dt: time step size
h: space step size
""" -> 
function transprob(x, y, dt, h, p_trunc = 0.45, c_trunc = 5) 
	if abs(y-x) < 3*sqrt(2*dt*log(1/dt))
		return exp(-(y-x)^2/(2*dt))/sqrt(2*pi*dt)*h
	else
		return 0
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
function pmatrix0(n::Int, h, T = 1, x0 = 0, lb = -3, p_trunc=0.5, c_trunc=1)
x_mesh = (g(T/n)-h/2):(-h):(lb)
l = length(x_mesh)
lb = x_mesh[end]
p0_vec = zeros(l)
	for j = 1:(l-1)
		p0_vec[j] = bbb(x0, x_mesh[j], 0, T/n)*transprob(x0, x_mesh[j], T/n, h, p_trunc, c_trunc)
	end
p0_vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, h, lb, p_trunc, c_trunc) 
return p0_vec
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
function pmatrix(i::Int, n::Int, h, T = 1, lb = -3, p_trunc=0.5, c_trunc=1)
jrange = (g(T*i/n)-h/2):(-h):(lb) # moving from i to i+1
krange = (g(T*(i+1)/n)-h/2):(-h):(lb)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*i/n, T*(i+1)/n)*transprob(jrange[j], krange[k], T/n, h, p_trunc, c_trunc)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n)*C(jrange[j], T/n, h, lb, p_trunc, c_trunc)
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
function BCP(n::Int, h, T = 1, x0 = 0, lb = -3, p_trunc=0.45, c_trunc=4)
    if (g(T) - lb < h) | (x0 > g(0))
        return 1
    end
	prob = transpose(pmatrix0(n, h, T, x0, lb, p_trunc, c_trunc))
	for i = 1:(n-1)
		prob = prob*pmatrix(i, n, h, T, lb, p_trunc, c_trunc)
	end
	return 1 - sum(prob)
	# return prob
end

