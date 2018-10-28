using PyPlot
using Distributions
using Statistics

# Daniels Boundary
function g(t)
	return 0.5 - t*log(0.25*(1 + sqrt( 1 + 8*exp(-1/t) ) ) )
end

# Exact non-BCP
function exact_limit(T = 1, theta = 1)
	if T == 0 
	  return 0
	end
	a1 = a2 = 1/2
	a = 1
	b1 = cdf(Normal(0 ,sqrt(T)),g(T)) 
	b2 = a1*cdf(Normal(theta,sqrt(T)),g(T)) 
	b3 = a2*cdf(Normal(2*theta,sqrt(T)),g(T)) 
	return (b1 - (b2 + b3)/a)
end


# Exact FDD BM generation
function wiener(n)
return cat(0,cumsum(randn(n))/sqrt(n), dims = 1)
end


# Calculates if a Brownian bridge passing through the endpoints
# X and X2 crosses a boundary g. Returns a vector of 1 if hit, 0 if not 
function mult(X)
X2 = -X # antithetic variable
n = length(X)-1
vec = zeros(n)
vec2 = zeros(n)
t_vec = 0:(1/n):1
	for i = 1:n
		if (X[i+1] < g(t_vec[i+1])) & (X[i] < g(t_vec[i]))
			vec[i] = 1 - exp( -2 * n * ( g(t_vec[i+1]) - X[i+1] ) * ( g(t_vec[i]) - X[i] ) )
		else
			vec[i] = 0
		end
		if (X2[i+1] < g(t_vec[i+1])) & (X2[i] < g(t_vec[i]))
			vec2[i] = 1 - exp( -2 * n * ( g(t_vec[i+1]) - X2[i+1] ) * ( g(t_vec[i]) - X2[i] ) )
		else
			vec2[i] = 0
		end
	end
   return mean([prod(vec), prod(vec2)])
end


# Difference in the BCPs of RW and Brownian motion
function monte_carlo(N,n)
vec = zeros(N)
	for i in 1:N
		X = wiener(n)
		vec[i] = mult(X) 
	end
	return mean(vec)
end

# monte carlo live stream
function monte_carlo_2(n = 10)
X = wiener(n)
runningmean = mult(X) 
i = 1
	while 1 == 1
		X = wiener(n)
		runningmean =  i/(i+1)*runningmean +  mult(X)/(i+1) 
		if i % 2500 == 0
			print("running mean: ", runningmean, " iteration: ", i ,"\r")
		end
		i +=1 
	end
	return runningmean
end


# Converge guide plot
function guideplot(N, s = 0.05)
	plt[:yscale]("log")
	plt[:xscale]("log")
	grid("on")
	plot(1:N, 1 ./((1:N).^0.5)*s, label = L"$1/\sqrt{n}$", ls = "--", color = "black")
	plot(1:N, 1 ./(1:N)*s  ,	     label = L"1/n",     ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^2)*s ,   label = L"1/n^2",   ls = "--", color = "black")
	plot(1:N, 1 ./((1:N).^4)*s ,   label = L"1/n^4",   ls=  "--", color = "black")
end


@doc """
    conv(M, B, delta) 

Plots convergence
M: n ranges from 1 : M
B: Number of Monte carlo variates
""" -> 
function conv_plot(M,B,delta)
xvec = 1:delta:M
vec = zeros(length(xvec))
i = 1
for x in xvec
	vec[i] = abs(monte_carlo(B,x) - exact_limit())
	i = i + 1
end
guideplot(M)
plot(xvec, vec,marker="o")
return vec
end

conv_plot(10,1000000,1)