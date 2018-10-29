using Distributions 

function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

function g(t, n)
ti = floor(t.*n)/n
m = n.*(g(ti + 1 ./n) - g(ti))
c = g(ti) - ti.*m
	return m.*t + c
end

function K_gen(N, n)
k = 1:N
h = 1/N
K = zeros(length(k),length(k))
function b(t)
	return (g(t,n) - g(0,n))/t
end
for j = 1:length(k)
	for i = 1:j
		t = j*h
		s = (2i-1)*h/2
		K[j,i] =  cdf(Normal(0,1), ( g(s,n) - g(t,n) )/sqrt(t-s)) + exp( -2*b(t)*( g(t,n) - g(s,n) - (t-s)*b(t) ) )*
			cdf(Normal(0,1), ( g(s,n) - g(t,n) + 2*(t-s)*b(t) )/sqrt(t-s))
	end
end
return K
end

function F_vec(N, n)
k = 1:N
h = 1/N
F = zeros(length(k))
function b(t)
	return (g(t,n) - g(0,n))/t
end
for i = 1:length(k)
	t = i*h
	F[i] = cdf(Normal(0,1), - g(t,n)/sqrt(t)) + exp(-2*b(t)*(g(t,n) - t*b(t)))*cdf(Normal(0,1), (-g(t,n) + 2t*b(t))/sqrt(t))
end
return F
end

function BCP_LD(N, n)
	return sum(K_gen(N, n)\F_vec(N, n))
end
