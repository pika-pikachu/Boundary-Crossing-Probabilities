using Distributions 

@doc """
	g(t) 

Returns the Daniel's boundary at time t. 
""" -> 
function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

@doc """
	exact_limit(T = 1) 

Returns the exact probability that a Brownian motion crosses Daniel's boundary
""" ->
function exact_limit(T = 1)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)), g(T)) 
	b2 = a1*cdf(Normal(1,sqrt(T)),g(T)) 
	b3 = a2*cdf(Normal(2,sqrt(T)),g(T)) 
	return 1 - (b1 - (b2 + b3)/a)
end

@doc """
	g(t, x0, n)

n-piecewise linear approximation of g(t)
n: Maximum number of boundary partitions
""" -> 
function g(t, x0, n = 0)
if n == 0
	return g(t) - x0
end
ti = floor(t.*n)/n
m = n.*(g(ti + 1 ./n) - g(ti))
c = g(ti) - ti.*m
	return m.*t + c
end


function K_gen(N, n = 0, T = 1, x0 = 0, t0 = 0)
h = (T-t0)/N
K = zeros(N,N)
function b(t)
	#return (g(t,x0,n) - g(t0,x0,n))/(t-t0)
	return 0
end
for j = 1:N
	for i = 1:j
		t = t0 + j*h
		s = t0 + (2i-1)*h/2
		K[j,i] =  cdf(Normal(0,1), ( g(s,x0,n) - g(t,x0,n) )/sqrt(t-s)) + exp( -2*b(t)*( g(t,x0,n) - g(s,x0,n) - (t-s)*b(t) ) )*
			cdf(Normal(0,1), ( g(s,x0,n) - g(t,x0,n) + 2*(t-s)*b(t) )/sqrt(t-s))
	end
end
return K
end

function F_vec(N, n = 0, T = 1, x0 = 0, t0 = 0)
h = (T-t0)/N
F = zeros(N)
function b(t)
	#return (g(t,x0,n) - g(t0,x0,n))/(t-t0)
	return 0
end
for i = 1:N
	t = t0 + i*h
	F[i] = cdf(Normal(0,1), -g(t,x0,n)/sqrt(t)) + exp(-2*b(t)*(g(t,x0,n) - t*b(t)))*cdf(Normal(0,1), (-g(t,x0,n) + 2t*b(t))/sqrt(t))
end
return F
end

@doc """
	BCP_LD(N, n, T, x0, t0)

Returns approximated boundary crossing probability (Loader & Deely 1987)
N: Number of partitions
n: Number of piecewise linear partitions. Set to 0 to keep the original boundary.
T: Terminal time
""" -> 
function BCP_LD(N, n = 0, T = 1, x0 = 0, t0 = 0)
	return sum(K_gen(N, n, T)\F_vec(N, n, T, x0, t0))
end
