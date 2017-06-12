function export_graph(results,NN)
cv = [0 0 0;...
    0 0 1;...
    0 0.5 0];
mk = {'o','^','v'};
ms = [7 8 8];
lw = 2;
xsize = 500;
ysize = 400;
nfont = 20;
lfont = 18;
xfont = 20;
baseline = 1;
logpmeasure = @(x,y) mean(x./y,1);
f1 = figure; clf
set(f1,'Position',[1400 200 xsize ysize]),
hold on;
grid on;
pp = [];
for k = 1:3
    p = plot(NN, mean(results{baseline}.tim ./ results{k}.tim, 1), '-','Marker',mk{k}, 'Color', cv(k,:),...
            'MarkerFaceColor',cv(k,:),'DisplayName', results{k}.name,'LineWidth',lw,'MarkerSize',ms(k));
    pp = [pp p];
end
set(gca,'FontSize',nfont, 'FontName','Arial');
lg = legend(pp);
set(lg, 'FontSize',lfont,'Location','Northwest');
ylabel('speed-up (vs. greedy)','FontSize',xfont);
xlabel('matrix dimension','FontSize',xfont);
xlim([min(NN) max(NN)]);
print('figure_speedup.pdf', '-dpdf');
f2 = figure; clf;
set(f2,'Position',[1400-xsize 200 xsize ysize]),
hold on;
grid on;
pp = [];
for k = 1:3
    p = plot(NN, logpmeasure(results{k}.logp, results{baseline}.logp) , '-','Marker',mk{k}, 'Color', cv(k,:),...
            'MarkerFaceColor',cv(k,:),'DisplayName', results{k}.name,'LineWidth',lw,'MarkerSize',ms(k));
    pp = [pp p];
end
set(gca,'FontSize',nfont, 'FontName','Arial');
lg = legend(pp);
set(lg, 'FontSize',lfont,'Location','Southeast');
ylabel('log prob. ratio (vs. greedy)','FontSize',16);
xlabel('matrix dimension','FontSize',16);
xlim([min(NN) max(NN)]);
print('figure_logprob.pdf', '-dpdf');
end
