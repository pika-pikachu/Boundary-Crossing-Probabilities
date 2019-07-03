n = 1000
h = 1/sqrt(n)


function g(t, theta = 1)
	if t == 0
	  return theta/2
	end
	a1 = a2 = 1/2
	a = 1
  	return theta/2 - t/theta*log(0.5*a1/a + sqrt( 1/4*(a1/a)^2 + (a2/a)*exp(-theta^2/t) ) )
end


function transprob(x, y, dt, h)
	return exp(-((y-x)/h)^2/2)/sqrt(2*pi)
end

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




function test(n,t)
h = 1/sqrt(n)
i = floor(n*t)
lb = -3
T = 1 
x0 = g(T*floor(n/2)/n) - floor(g(T*floor(n/2)/n)/h)*h
# jrange = (g(T*i/n)-h):(-h):(lb) # moving from i to i+1
# krange1 = (g(T*(i+1)/n)-h):(-h):(-12)
krange1 = (g(T*(i+1)/n)):(-h):(-12)
krange2 = reverse((g(T*(i+1)/n)+1):(h):(12))

mesh = cat(krange2,krange1,dims = 1)

vector = zeros(length(mesh))
ci = constCi(i,n,h)
for i = 1:length(vector)
	vector[i] = transprob(x0,mesh[i],1/n,h)
end
p = vector/sum(vector)
m1 = mesh'*vector # first moment
m2 = (mesh.^2)'*vector # second moment 
m3 = (abs.(mesh).^3)'*vector # abs third moment
v = m2 - m1^2 # variance
sd = sqrt(v) # std dev
# plot(mesh, vector)
return [sum(p),m1,m3,abs(1/n - sd)^2]
end

function test2(n,t)
h = 1/sqrt(n)
i = floor(n*t)
lb = -3
T = 1 
x0 = g(T*floor(n/2)/n) - floor(g(T*floor(n/2)/n)/h)*h
# jrange = (g(T*i/n)-h):(-h):(lb) # moving from i to i+1
# krange1 = (g(T*(i+1)/n)-h):(-h):(-12)
krange1 = (g(T*(i+1)/n)):(-h):(-12)
krange2 = reverse((g(T*(i+1)/n)+1):(h):(12))

mesh = cat(krange2,krange1,dims = 1)

vector = zeros(length(mesh))
ci = constCi(i,n,h)
for i = 1:length(vector)
	vector[i] = transprob(x0,mesh[i],1/n,h)
end
p = vector/ci
m1 = mesh'*vector
m2 = (mesh.^2)'*vector 
m3 = (abs(mesh).^3)'*vector 
v = m2 - m1^2
plot(mesh, vector)
return [sum(p),m1,m3,v,abs(1/n - v)]
end


N = 1000
m1_vector = zeros(N)
m3_vector = zeros(N)
sd_vector = zeros(N)
for i in 1:N
m1_j_vector = zeros(i)
m3_j_vector = zeros(i)
sd_j_vector = zeros(i)
	for j in 1:i
		x = test(i,j/i)
		m1_j_vector[j] = x[2] 
		m3_j_vector[j] = x[3] 
		sd_j_vector[j] = x[4] 
	end
	m1_vector[i] = sum(m1_j_vector.^2)/i
	m3_vector[i] = sum(m3_j_vector)
	sd_vector[i] = sum(sd_j_vector)/i
end




plot(m1_vector)
plot(sd_vector)
plot(m3_vector)
plt[:xscale]("log")
plt[:yscale]("log")