function test_greedy
varying_dimension
end

function varying_dimension
w_m = 0.01;
b = 0.2;
ncluster = 5;
nrank = 20;
batch =10;
nsample = 50;
m = 20;
n = 15;

NN = (1:5) * 1000;

niter = 1;
nalgo = 3;
results = cell(1,nalgo);
for k = 1 : nalgo
    results{k}.tim = zeros(niter, length(NN));
    results{k}.logp = zeros(niter,length(NN));
    results{k}.name = '';
    results{k}.X = cell(niter, length(NN));
end

for t = 1 : niter
for i = 1 : length(NN)
    N = NN(i);
    L_kernel = synth_kernel(N,w_m,b);
    emin = 0.001;
    L_kernel = L_kernel + emin * speye(N);
    emax = eigs(L_kernel,1,'LA');
    
    j = 1;
    tic
    results{j}.X{t,i} = greedy_lazy(L_kernel);
    results{j}.tim(t,i) = toc;
    results{j}.logp(t,i) = logdet(L_kernel(results{j}.X{t,i},results{j}.X{t,i}));
    results{j}.name = 'Greey-Lazy';
    
    j = 2;
    tic
    results{j}.X{t,i} = greedy_alg1(L_kernel,ncluster,nrank);
    results{j}.tim(t,i) = toc;
    results{j}.logp(t,i) = logdet(L_kernel(results{j}.X{t,i},results{j}.X{t,i}));
    results{j}.name = 'Algorithm 1';

    j = 3;
    tic
    results{j}.X{t,i} = greedy_alg2(L_kernel,batch,nsample,ncluster,nrank,emax,emin,m,n);
    results{j}.tim(t,i) = toc;
    results{j}.logp(t,i) = logdet(L_kernel(results{j}.X{t,i},results{j}.X{t,i}));
    results{j}.name = 'Algorithm 2';
    
    fprintf('Dimension = %d with %d-th iteration\n', N, t);
    print_info(results,t,i);
end
end
export_graph(results,NN);
end

function print_info(results_ori,t,i)
len = 0;
idx = [];
for j = 1 : length(results_ori)
    if ~isempty(results_ori{j}.name)
        len = len + 1;
        idx = [idx j];
    end
end
base_line_idx = find(idx==1);
results = cell(1,len);
len = 0;
for j = 1 : length(results_ori)
    if ~isempty(results_ori{j}.name)
        len = len + 1;
        results{len} = results_ori{j};
    end
end
fmt = '---------------';
fprintf('%-15s|','');
for k = 1 : length(results)
    fprintf('%-15s',results{k}.name);
    fmt = [fmt '---------------'];
end
fmt = [fmt '\n'];
fprintf('\n');
fprintf('%-15s|','time');
for k = 1 : length(results)
    fprintf('%-15.2f',results{k}.tim(t,i));
end
fprintf('\n');
fprintf('%-15s|','speedup');
for k = 1 : length(results)
    fprintf('%-15.2f',results{base_line_idx}.tim(t,i)/results{k}.tim(t,i));
end
fprintf('\n');
fprintf(fmt);
fprintf('%-15s|','logdet');
for k = 1 : length(results)
    fprintf('%-15.2f',results{k}.logp(t,i));
end
fprintf('\n');
fprintf('%-15s|','diff-logp');
for k = 1 : length(results)
    fprintf('%-15.4f',results{k}.logp(t,i) - results{base_line_idx}.logp(t,i));
end
fprintf('\n');
fprintf('%-15s|','logp-ratio');
for k = 1 : length(results)
    fprintf('%-15.4f',results{k}.logp(t,i) ./ results{base_line_idx}.logp(t,i));
end
fprintf('\n');
fprintf('%-15s|','num-ele');
for k = 1 : length(results)
    fprintf('%-15.2f',length(results{k}.X{t,i}));
end
fprintf('\n');
fprintf(fmt);
end

function L_kernel = synth_kernel(N,w_m,b)
m = randn(N,1);
S = randn(N,N);
S = S ./ repmat(sqrt(sum(S.^2,2)),1,N);

L_scaled = (S*S');

M = spdiags(sqrt(exp(w_m * m + b)),0,N,N);
L_kernel = M * (L_scaled * M);
L_kernel = (L_kernel + L_kernel')/2;
end
