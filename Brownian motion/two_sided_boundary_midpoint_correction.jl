using Distributions

@doc """
    gU(t)

Upper boundary. 
Reference: Lerche. H, "Boundary Crossing of Brownian motion
Its Relation to the Law of the Iterated Logarithm and to Sequential Analysis"
Chapter I Example 3 p28 (1986)
""" -> 
function gU(t, a=2, theta =3)
    if t == 0
        return theta/2
    else
        return t/theta*acosh(a*exp(theta^2/(2t)))
    end
end

@doc """
    gL(t)

Lower boundary.
Reference: Lerche. H, "Boundary Crossing of Brownian motion
Its Relation to the Law of the Iterated Logarithm and to Sequential Analysis"
Chapter I Example 3 p28 (1986)
""" -> 
function gL(t, a=2, theta = 3)
    if t == 0
        return -theta/2
    else
        return -t/theta*acosh(a*exp(theta^2/(2t)))
    end
end

@doc """
    exact_limit(a = 2 , theta = 3) 
Returns the exact probability that a Brownian motion crosses the two sided boundary gU and gL
Reference: Lerche. H, "Boundary Crossing of Brownian motion
Its Relation to the Law of the Iterated Logarithm and to Sequential Analysis"
Chapter I Example 3 p28 (1986)
""" -> 
function exact_limit(a = 2 , theta = 3,T=1, eps = 1/100)
if T < eps
    return 1
end
b1 = cdf(Normal(0,     sqrt(T)),gU(T)) - cdf(Normal(0,     sqrt(T)),gL(T))
b2 = cdf(Normal(theta ,sqrt(T)),gU(T)) - cdf(Normal(theta, sqrt(T)),gL(T))
b3 = cdf(Normal(-theta,sqrt(T)),gU(T)) - cdf(Normal(-theta,sqrt(T)),gL(T))
return b1 - (b2 + b3)/2/a
end


@doc """
    bbb(x0, x1, t0, t2) 
Returns the exact probability that a Brownian bridge starting at x0 and ending at x1 survives 
a piecewise linear boundary approximation of g(t):
g(t), t0  <= t <= t2
""" -> 
function bbb(x0,x1,t0,t1)
    return 1 - exp(-2(gU(t1) - x1)*(gU(t0)-x0)/(t1-t0)) - exp(-2(gL(t1) - x1)*(gL(t0)-x0)/(t1-t0))
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
    pmatrix0(n, h)
Returns the transition probability matrix of the Markov chain approximation of Brownian motion 
from time 0 to time 1/n
n: number of time partitions
h: space step size
""" -> 
function pmatrix0(n::Int, h)
m = ceil(Int64, (gU(1/n) - gL(1/n))*n) # space partitioning
domain = range( gU(1/n)-h/2, stop = gL(1/n) + h/2, length = m)
l = length(domain)
h = abs(domain[1] - domain[2])
vec = zeros(l)
    for j = 1:l
        vec[j] = transprob(0, domain[j], 1/n, h)*bbb(0, domain[j], 0, 1/n)
    end
return vec
end

@doc """
    pmatrix(i, n, h)
Returns the transition probability matrix of the Markov chain approximation of Brownian motion
from time i/n to time (i+1)/n
i: ith time partition 
n: number of time partitions
h: space step size
""" -> 
function pmatrix(i::Int, n::Int, h)
m1 = ceil(Int64, (gU(i/n) - gL(i/n))*n)
m2 = ceil(Int64, (gU((i+1)/n) - gL((i+1)/n))*n)
jrange = range( gU(i/n) - h/2, stop = gL(i/n) + h/2, length = m1) # moving from i to i+1
krange = range( gU((i+1)/n) - h/2, stop = gL((i+1)/n) + h/2, length = m2) # moving from i to i+1
h = abs(krange[1] - krange[2])
M = zeros(length(jrange),length(krange))
    for j = 1:length(jrange)
        for k = 1:length(krange)
            M[j, k] = transprob(jrange[j], krange[k], 1/n, h)*bbb(jrange[j], krange[k], i/n, (i+1)/n)
        end
    end
return M
end

@doc """
    BCP(n::Int, h)
Returns the approximated boundary crossing probability
n: number of time partitions
h: space step size
""" -> 
function BCP(n::Int, h)
    prob = transpose(pmatrix0(n, h))
    for i = 1:(n-1)
        prob = prob*pmatrix(i, n, h)
    end
    return 1 - sum(prob)
end




