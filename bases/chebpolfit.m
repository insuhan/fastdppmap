function c = chebpolfit(fname,n)
x = cos(((0:n)'+0.5)*pi/(n+1));
y = feval(fname,x);
T = [zeros(n+1,1) ones(n+1,1)];
c = [sum(y)/(n+1) zeros(1,n)];
a = 1;
for k = 2 : n+1
    T = [T(:,2) a*x.*T(:,2)-T(:,1)];
    c(k) = (y'*T(:,2))*2/(n+1);
    a = 2;
end
end
