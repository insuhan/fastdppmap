function ld = cheb_logdet(AX,emin,emax,m,n,vv)
delta = emin / (emin + emax);
r = 2/(1-2*delta);
b = -1/(1-2*delta);

AX_hat = (AX/(emin+emax)).*r + b*speye(size(AX));

f = @(x) log(x);
g = @(x) ((1-2*delta)/2).*x+0.5;
h = @(x) f(g(x));
c = chebpolfit(h,n);

w0 = vv;
w1 = AX_hat*vv;
u = c(1)*w0 + c(2)*w1;
for j = 2 : n
    ww = 2*(AX_hat)*w1 - w0;
    u = u + c(j+1)*ww;
    w0 = w1;
    w1 = ww;
end
ld = sum(sum(u.*vv))/m + size(AX,1)*log(emin + emax);
end