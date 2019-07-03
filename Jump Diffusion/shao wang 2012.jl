# Bugs: When lambda =2 , sometimes it returns NaN


using Distributions

@doc """
    ind(a,b)

Indicator function, returns 1 if a < b and 0 elsewise.
"""->
function ind(a,b)
    if a < b
        return 1
    else
        return 0
    end
end


# Boundary function

function g(t)
    return 1
    # return -0.5*t + 1.5
    # return exp(-t)
end

@doc """
    b(t_i,k,u_vec_k,h_vec_k)

Boundary function that shifts with the jumps.
plot(0:0.01:1,map(b,0:0.01:1))
"""->
function b(t_i,k,u_vec_k,h_vec_k)
adj = 0
if k == 0
    return g(t_i)
    #return sqrt(1+t_i)
else
    for i = 1:k
        if t_i > u_vec_k[i]
            adj += h_vec_k[i]
        end    
    end
    return g(t_i) - adj       
    # return sqrt(1+t_i) - adj       
end
end

@doc """
    integrand(b,b_vec_m,t_vec,u_vec_k,h_vec_k,m,k)

BCP for a given path.
plot(0:0.01:1,map(b,0:0.01:1))
"""->
function integrand(b,b_vec_m,t_vec,u_vec_k,h_vec_k,m,k)
prod = 1
for i = 2:(m+1)
    prod = (1 - exp( -2(b( t_vec[i-1], k, u_vec_k, h_vec_k ) - b_vec_m[i-1])*(b( t_vec[i], k, u_vec_k, h_vec_k ) - b_vec_m[i])/( t_vec[i] - t_vec[i-1]) ) )* prod * ind(b_vec_m[i], b(t_vec[i], k, u_vec_k, h_vec_k) )
end
return prod
end

@doc """
   randDE(eta1 = 0.1, eta2=0.15)

Generates a RV ~ Double exponential
plt[:hist](randDE(270,0.1,0.15),bins=90)
"""->
function randDE(n = 1, eta1 = 0.1, eta2 = 0.15)
vec = zeros(n)
    for i = 1:n
        B = rand(Bernoulli(),1)    
        if B[1]  ==1
            vec[i] = rand(Exponential(eta1),1)[1]
        else
            vec[i] = -rand(Exponential(eta2),1)[1]
        end
    end
return vec
end


@doc """
    monte_carlo(lambda)

Returns BCP for a given jump height and jump time. Uses `lambda` jumps.

monte_carlo(3)
"""->
function monte_carlo(lambda)
n = 32
k = rand(Poisson(lambda))
m = k + n
u_vec_k = sort(rand(k))
#h_vec_k = randDE(k, 0.1, 0.15)
#h_vec_k = 0.5*ones(k)
# h_vec_k = randDE(k)
h_vec_k = rand(Exponential(0.15), k)
#h_vec_k = (rand(Bernoulli(0.5),k)-0.5)*0.3
b_vec_m = zeros(m+1) # including at t_0 = 0
s_vec = (1/n):(1/n):1 # ignore s_0 = 0
t_vec = sort(cat(s_vec,u_vec_k,0,dims=1))
# t_vec = cat(1,0,t_vec,dims=1)
for i = 2:(m+1)
    b_vec_m[i] = b_vec_m[i-1] + sqrt(t_vec[i] - t_vec[i-1])*rand(Normal())
end
#c = rand(3,1)
#plot(t_vec,b_vec_m,color = c)
#plot(0:0.01:1,map(b,0:0.01:1),color = c)
return max(integrand(b,b_vec_m,t_vec,u_vec_k,h_vec_k,m,k),0)
end


@doc """
    serial_monte_carlo(lambda, N)

Monte_carlo integration.

lambda: controls rate of jumps
@time serial_monte_carlo(1, 10^5)
For jump size 0.5, we get [0.279455, 0.000978008, 0.720545]
"""->
function serial_monte_carlo(lambda,N::Int)
vec = zeros(N)
    for i=1:N
        vec[i] = monte_carlo(lambda)
    end
    return [mean(vec), std(vec)/(sqrt(N)-1), 1- mean(vec)]
end

