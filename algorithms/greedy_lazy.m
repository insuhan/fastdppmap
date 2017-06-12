function X = greedy_lazy(A)
N = size(A,1);
deltas = inf * ones(1,N);
X = [];
Y = 1:N;
i = 0;
cg_iter = 30;
while 1
    i = i + 1;
    bestimprov = 0;
    [~,order] = sort(deltas,'descend');
    for test = order
        if ismember(Y(test), X)
            deltas(test) = -inf;
        end
        if deltas(test) >= bestimprov
            improv = logdet_margin_cg(A,X,Y(test),cg_iter);
            deltas(test) = improv;
            bestimprov = max(bestimprov, improv);
        elseif deltas(test) > -inf
           break;
        end
    end
    argmax = find(deltas == max(deltas),1);
    if deltas(argmax) >= 0
        X = [X Y(argmax)];
    else
        break;
    end
end
end


