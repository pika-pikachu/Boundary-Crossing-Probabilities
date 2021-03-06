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
    # return 1
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

function C_final(x, dt, h, lb)
range = lb:(-h):(-12)
l = length(range)
vec = zeros(l)
for j = 1:l
	vec[j] = transprob_final(x, range[j], dt, h)
end
return sum(vec)
end

# Boundary dependent Scaling constant 
function constCi(i, n, h, T=1)
range1 = ((g(T*(i+1)/n) -g(T*i/n))/h):1:12
range2 = ((g(T*(i+1)/n) -g(T*i/n))/h - 1):(-1):-12
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
	for i in 1:length(range1)
		vec1[i] = exp(-range1[i]^2/2)
	end
	for i in 1:length(range2)
		vec2[i] = exp(-range2[i]^2/2)
	end
	return (sum(vec1) + sum(vec2))/sqrt(2*pi)
end

function constC0(n, h, T=1)
range1 = (g(T/n)/h):1:12
range2 = (g(T/n)/h - 1):(-1):-12
vec1 = zeros(length(range1))
vec2 = zeros(length(range2))
	for i in 1:length(range1)
		vec1[i] = exp(-range1[i]^2/2)
	end
	for i in 1:length(range2)
		vec2[i] = exp(-range2[i]^2/2)
	end
	return (sum(vec1) + sum(vec2))/sqrt(2*pi)
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
	return exp(-((y-x)/h)^2/2)/sqrt(2*pi)
end

function transprob_final(x, y, dt, h)
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
# range = (g(T/n)-h/2):(-h):(lb)
range = (g(T/n)-h):(-h):(lb)
c0 = constC0(n, h, 1)
l = length(range)
lb = range[end]
vec = zeros(l)
	for j = 1:(l-1)
		vec[j] = bbb(x0, range[j], 0, T/n)*transprob(x0, range[j], T/n, h)
	end
vec[end] = bbb(x0, lb, 0, T/n)*C(x0, T/n, h, lb)
return vec/c0
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
# jrange = (g(T*i/n)-h/2):(-h):(lb) # moving from i to i+1
# krange = (g(T*(i+1)/n)-h/2):(-h):(lb)
jrange = (g(T*i/n)-h):(-h):(lb) # moving from i to i+1
krange = (g(T*(i+1)/n)-h):(-h):(lb)
ci = constCi(i, n, h, 1)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*i/n, T*(i+1)/n)*transprob(jrange[j], krange[k], T/n, h)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*i/n, T*(i+1)/n)*C(jrange[j], T/n, h, lb)
	end
M = M/ci 	
M[length(jrange), length(krange)] = 1
return M
end


function pmatrix_end(n::Int, h, T = 1, lb = -3)
h2 = 1/n
# jrange = (g(T*(n-1)/n)-h/2):(-h):(lb) 
# krange = (g(T)-h2/2):(-h2):(lb)
jrange = (g(T*(n-1)/n)-h):(-h):(lb) 
krange = (g(T)-h2/2):(-h2):(lb)
# krange = (g(T)-h2/2):(-h2):(-3)
# cn = constCi(n-1, n, h, 1)
# cn = constCi(n, n, h, 1)
lb = krange[length(krange)]
M = zeros(length(jrange),length(krange))
	for j = 1:(length(jrange)-1)
		for k = 1:(length(krange)-1)
			M[j, k] = bbb(jrange[j], krange[k], T*(n-1)/n, T)*transprob_final(jrange[j], krange[k], T/n, h2)
		end
		M[j, length(krange)] = bbb(jrange[j], lb, T*(n-1)/n, T)*C_final(jrange[j], T/n, h2, lb)
	end
# M = M/cn
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


function BCP_vec(n::Int, h, T = 1, x0 = 0, lb = -3, c = 1)
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
	return prob
end

# function RichardsonExtrap(N, lb = -3, p = 2, q = 2, l = 1, x0 = 0, T = 1)
# M = zeros(N, N)
# 	function A(n)
# 		return BCP(n, 1/sqrt(n), T, x0, lb)
# 	end
# M[1, 1] = A(2^0)
# for j = 2:N
# 	M[j,1] = A(q^(j-1))
# 	for k = 2:j
# 		M[j, k] = (q^((k-1)*p)*M[j, k-1] - M[j-1, k-1])/(q^((k-1)*p)-1)
# 	end
# end
# return M
# end

function figureplot(n)
h = 1/n
mesh = (g(1)-h/2):(-h):-3
x = BCP_vec(n,1)
y = cat(x[1:(end-1)]/h, x[end],dims =1)
plot(mesh,y)
end