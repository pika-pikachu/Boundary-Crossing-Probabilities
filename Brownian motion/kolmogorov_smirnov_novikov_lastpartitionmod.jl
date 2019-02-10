using Distributions

@doc """
    gU(t)

Upper boundary
""" -> 
function gU(t)
    return 3*sqrt(t)
end

@doc """
    gL(t)

Lower boundary
""" -> 
function gL(t)
	return -3*sqrt(t)
end

# @doc """
#     bbb(x0, x1, t0, t2) 

# Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
# a piecewise linear boundary approximation of g(t):
# g(t), t0  <= t <= t2
# """ -> 
# function bbb(x0,x1,t0,t1)
# 	return 1 - exp(-2(gU(t1) - x1)*(gU(t0)-x0)/(t1-t0)) - exp(-2(gL(t1) - x1)*(gL(t0)-x0)/(t1-t0))
# end

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

function bbb(x0, x1, t0, t2)
        t1 = (t0 + t2)/2 #half point between times
        b1_1 = (gU(t1) - gU(t0))/(t1 - t0) 
        a1_1 = gU(t0) - x0
        b2_1 = (gU(t2) - gU(t1))/(t2 - t1)
        a2_1 = gU(t2) - b2_1*(t2 - t0) - x0
        b1_2 = -(gL(t1) - gL(t0))/(t1 - t0) 
        a1_2 = x0 - gL(t0)
        b2_2 = -(gL(t2) - gL(t1))/(t2 - t1)
        a2_2 = -gL(t2) - b2_2*(t2 - t0) + x0
        return 1 - J(a1_1, b1_1, a2_1, b2_1, t1 - t0, t2 - t0, x1 - x0) - J(a1_2, b1_2, a2_2, b2_2, t1 - t0, t2 - t0, x0-x1)
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
    pmatrix0(t0, m)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time t0
t0:
m: number of space partitions
""" -> 
function pmatrix0(t0, m::Int)
h = (gU(t0) - gL(t0))/m
domain = range(gU(t0)-h/2, stop = gL(t0)+h/2, length = m)
d = domain[1] - domain[2]
l = length(domain)
vec = zeros(l)
	for j = 1:l
		vec[j] = transprob(0, domain[j], t0, d)
	end
return vec
end

@doc """
    pmatrix1(n, m, t, t0)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion
from time t0 to time 2/n
n: number of time partitions
m: number of space partitions
t: time_mesh
t0: 
""" -> 
function pmatrix1(n::Int, m::Int, t, t0)
    h = (gU(t0) - gL(t0))/m
    jrange = range(gU(t0)-h/2, stop = gL(t0)+h/2, length = m)
    hk = (gU(t[2]) - gL(t[2]))/m
    krange = range(gU(t[2])-hk/2, stop = gL(t[2])+hk/2, length = m)
    d = krange[1] - krange[2]
    M = zeros(length(jrange),length(krange))
    for j = 1:length(jrange)
        for k = 1:length(krange)
            M[j, k] = transprob(jrange[j], krange[k], t[2]-t0, d)*bbb(jrange[j], krange[k], t0, t[2])
        end
    end
return M
end

@doc """
    pmatrix(i, n, m, t, t0)

Returns the transition probability matrix of the Markov chain approximation of Brownian motion
from time t0 to time 2/n
i: ith time partition 
n: number of time partitions
m: number of space partitions
t: time_mesh
t0: 
""" -> 
function pmatrix(i::Int, n::Int, m::Int, t, t0)
    h = (gU(t[i]) - gL(t[i]))/m
    jrange = range(gU(t[i])-h/2, stop = gL(t[i])+h/2, length = m)
    hk = (gU(t[i+1]) - gL(t[i+1]))/m
    krange = range(gU(t[i+1])-hk/2, stop = gL(t[i+1])+hk/2, length = m)
    d = krange[1] - krange[2]
    M = zeros(length(jrange),length(krange))
    for j = 1:length(jrange)
        for k = 1:length(krange)
            M[j, k] = transprob(jrange[j], krange[k], t[i+1]-t[i], d)*bbb(jrange[j], krange[k], t[i], t[i+1])
        end
    end
return M
end
 


function pmatrix_end(n::Int, m::Int, t)
    h = (gU(t[n-1]) - gL(t[n-1]))/m
    h2 = 3/n^2 
    m2 = ceil(Int64, (gU(1) - gL(1))/h2)
    jrange = range( gU(t[n-1]) - h/2, stop = gL(t[n-1]) + h/2, length = m) # moving from i to i+1
    krange = range( gU(1) - h2/2, stop = gL(1) + h2/2, length = m2) # moving from i to i+1
    d = krange[1] - krange[2] 
    M = zeros(length(jrange),length(krange))
    for j = 1:length(jrange)
        for k = 1:length(krange)
            M[j, k] = bbb(jrange[j], krange[k], (n-1)/n, 1)*transprob(jrange[j], krange[k], t[n]-t[n-1], d)
        end
    end
return M
end



@doc """
    time_mesh(n, t0, p = 2)

Non linear time mesh
n: number of time partitions
t0: 
p: polynomial powerfor time spacing
""" -> 
function time_mesh(n::Int, t0, p = 2)
    s = 1/(1-t0)^(p-1)
    vec = s*(Array(range(t0, stop = 1, length = n)) - ones(n)*t0).^p + ones(n)*t0
    return vec
end

@doc """
    BCP(n::Int, m::Int)

Returns the approximated boundary crossing probability
n: number of time partitions
m: number of space partitions
""" -> 
function BCP(n::Int, m::Int)
    a = 0.01
    b = 0.99
	t0 = a*(1-b)/(1-a)/b
    t = time_mesh(n, t0)
	prob = transpose(pmatrix0(t0, m))
	prob = prob*pmatrix1(n, m, t, t0)
	for i = 2:(n-2)
		prob = prob*pmatrix(i, n, m, t, t0)
	end
    prob = prob*pmatrix_end(n, m, t)
	return 1- sum(prob)
end

@doc """
    RichardsonExtrap(a, n::Int, burnin, p, q)

Applies Richardson's extrapolation to the BCP algorithm
a: scalar multiple of the n to m relationship
n: number of time partitions
m: number of space partitions
p: error order
q: increment of error powers. E = sum_{i=1}^n a_i h^{p + i q}

@time RichardsonExtrap(3,3,32)
""" -> 
function RichardsonExtrap(a, N, burnin = 32, p = 2, q = 2)
M = zeros(N, N)
    function A(n)
        return BCP(a*n, n)
    end
M[1, 1] = A(2^0+burnin)
for j = 2:N
    M[j, 1] = A(q^(j-1)+burnin)
    for k = 2:j
        M[j, k] =  (q^((k-1)*p)*M[j, k-1] - M[j-1, k-1])/(q^((k-1)*p)-1)
    end
end
return M
end

 

