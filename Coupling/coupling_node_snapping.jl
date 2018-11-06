using PyPlot
using Statistics
using Distributions

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


# survival probability
# function monte_carlo(N,n)
# vec = zeros(N)
# 	for i in 1:N
# 		vec[i] = mult(wiener(n))
# 	end
# 	return mean(vec)
# end

# Generating the RW from the Brownian motion
# Snap to nearest node. Gives n^3/2 convergence.
function wiener2mc(X)
vec = zeros(length(X))
n = length(X) - 1
h = 1/n
	for i in 1:length(X)
		vec[i] = round(X[i]/h)*h
	end
return vec
end


# function wiener2mc(X)
# vec = zeros(length(X))
# n = length(X) - 1
# h = 1/n
# t_vec = 0:(1/n):1
# 	for i in 1:length(X)
# 		vec[i] = g(t_vec[i]) - round((g(t_vec[i]) - X[i])/h)*h
# 	end
# return vec
# end


# Calculates if a Brownian bridge passing through the endpoints
# X and X2 crosses a boundary g. 
function mult(X)
X2 = wiener2mc(X)
n = length(X)-1
t_vec = 0:(1/n):1
vec2 = zeros(n)
	for i = 1:n
		if (X2[i+1] < g(t_vec[i+1])) & (X2[i] < g(t_vec[i]))
			vec2[i] = 1 - exp( -2 * n * ( g(t_vec[i+1]) - X2[i+1] ) * ( g(t_vec[i]) - X2[i] ) )
		else
			vec2[i] = 0
		end
	end
   return prod(vec2)
end

# Difference in the BCPs of RW and Brownian motion
function monte_carlo(N,n)
vec = zeros(N)
	for i in 1:N
		X = wiener(n)
		Z = mult(X)
		vec[i] = Z
	end
	return abs(mean(vec) - exact_limit())
end

 # monte_carlo2(100,40)

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
    dyn_conv_plot(n)

Plots convergence dynamically.
Careful, has an unfinishing monte carlo loop
dyn_conv_plot(10)
""" -> 
function dyn_conv_plot(n)
i = 1
bcp_vec= zeros(n)
while 1 == 1
	for j in 1:n
		bcp_vec[j] = bcp_vec[j]*i/(i+1) +  mult(wiener(j))/(i+1)
	end
	if i % 10000 == 0
		# fitting linear model
		error_vec = abs.(bcp_vec - ones(n) .* exact_limit())
		x = log.(1:n)
		y = log.(error_vec)

		beta_1 = cov(x,y)/var(x)
		# beta_0 = mean(y) - beta_1*mean(x)
		beta_0 = y[1] - beta_1*x[1]

		y_fit = exp.( beta_0*ones(n) + beta_1*x )

		plt[:cla]()
		guideplot(n)
		plot(1:n, error_vec, marker = "o")	
		plot(1:n, y_fit)	
		plt[:pause](0.0001)
		print("iteration: ", i , " slope: ", beta_1, "\r")
	end
	i += 1
end
end



# Convergence rate plot

function conv_plot(M,B,delta=1)
xvec = 1:delta:M
vec = zeros(length(xvec))
i = 1
for x in xvec
	vec[i] = monte_carlo(B,x)
	i = i + 1
end
guideplot(M)
plot(xvec, vec,marker="o")
return vec
end



