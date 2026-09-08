% Dissociable roles of human frontal eye fields and early visual cortex in presaccadic attention
% Nina M. Hanning, Antonio Fernandez, & Marisa Carrasco
%
% analysis script / figure generation
%
% 2023-07-19

% Figure 2a
% Figure 2b
% Figure 3 
% Figure S1a
% Figure S1b
% Figure S2

clear all;
close all;
clc;

% Colors
colPSA  = [110, 98,168]./255;
colEXO  = [ 53,163,182]./255;
colENDO = [236,163,108]./255;

colV    = [110, 98,168]./255;
colN    = [116,116,116]./255;
colI    = [ 95,183,149]./255;

colTMS  = [255,215,  0]./255;
colSym  = [255,255,255]./255;
colDiff = mean([colTMS;colSym],1);

colStim = [166,166,166]./255;
colNot  = [ 89, 89, 89]./255;


%% Figure 2a
load('fig2_psa.mat');

fig2a = figure; title('V1/V2 TMS - presaccadic attention');
hold on;

% fit individual subject's data
n       = size(fig2_psa.valid_T,1);
ncond   = 6;
cLevels = logspace(log10(0.02),log10(0.85),7);
cLevels = cLevels(1,[1,3:end])';

%transform the values to log and transform back after fitting
xx = log10(logspace(log10(0.01), log10(cLevels(end)), 1000)*100); % transform to log the interpolation
logContrast = log10(cLevels*100); % transform to log

ub = [5,logContrast(5),5];      % upper bound on params
lb = [5,logContrast(1),0.3];    % lower bound on params
x0 = [5,logContrast(3),2];      % starting point

% options for fmincom
opt = optimoptions('fmincon', 'Algorithm', 'interior-point','Display','off', 'MaxIter', 5000);

slope = nan(ncond,n);
c50   = nan(ncond,n);
dmax  = nan(ncond,n);

S = []; fit = []; interp_f = [];
for ii = 1:n % loop through observers

    S = [fig2_psa.valid_T(ii,:)',fig2_psa.neutral_T(ii,:)',fig2_psa.invalid_T(ii,:)',...
         fig2_psa.valid_n(ii,:)',fig2_psa.neutral_n(ii,:)',fig2_psa.invalid_n(ii,:)'];
    
    for jj = 1:size(S,2) % loop through conditions
        
        data = S(:,jj);
        
        f = @(par)nakaRushton(logContrast,par,data);
        [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
        
        [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
        [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit

        % store params
        slope(jj,ii) = par(1);
        c50(jj,ii)   = 10.^par(2); % convert back to contrast 
        dmax(jj,ii)  = par(3);
    end

end

% symbolic group average fit
S = [mean(fig2_psa.valid_T,1)',mean(fig2_psa.neutral_T,1)',mean(fig2_psa.invalid_T,1)',...
     mean(fig2_psa.valid_n,1)',mean(fig2_psa.neutral_n,1)',mean(fig2_psa.invalid_n,1)'];

av_slope = nan(size(S,2),1);
av_c50   = nan(size(S,2),1);
av_dmax  = nan(size(S,2),1);

fit = []; interp_f = [];
for jj = 1:size(S,2) % loop through conditions
    
    data = S(:,jj);
    
    f = @(par)nakaRushton(logContrast,par,data);
    [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
    
    [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
    [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit
    
    % store params
    av_slope(jj,1) = par(1);
    av_c50(jj,1)   = 10.^par(2); % convert back to contrast
    av_dmax(jj,1)  = par(3);
end

antilogC = 10.^(logContrast);
antixx   = 10.^(xx);

mk_size = 7;
line_w  = 2;

condCol_vec    = [colV;colN;colI;colV;colN;colI];
condTMSCol_vec = [colTMS;colTMS;colTMS;colSym;colSym;colSym];

for i = 1:3
    plot([0,150],[i,i],'k-'); hold on;
end

% plot interpolated fits & data points
for i = 1:3
    semilogx(antixx,interp_f(:,i),'-','color',condCol_vec(i,:),'linewidth',line_w);
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end
for i = 4:6
    semilogx(antixx,interp_f(:,i),'-.','color',condCol_vec(i,:),'linewidth',line_w); hold on;
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end 

% dmax
for i = 1:3
    plot([145,145],[mean(dmax(i,:),2) - std(dmax(i,:))./sqrt(n-1), mean(dmax(i,:),2) + std(dmax(i,:))./sqrt(n-1)],'Color',condCol_vec(i,:),'linewidth',line_w);
    plot([120,120],[mean(dmax(i+3,:),2) - std(dmax(i+3,:))./sqrt(n-1), mean(dmax(i+3,:),2) + std(dmax(i+3,:))./sqrt(n-1)],'Color',condCol_vec(i+3,:),'linewidth',line_w);
    
    scatter([145],mean(dmax(i,:),2),80,'MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:));
    scatter([120],mean(dmax(i+3,:),2),80,'MarkerEdgeColor',condCol_vec(i+3,:),'MarkerFaceColor',condTMSCol_vec(i+3,:));
end


set(gca,'Ylim',[0 3.75],'YTick',0:3,'Xlim',[0 150],'XTick',round(cLevels*100)');
xlabel('contrast (%)'); ylabel('visual sensitivity (d'')');
set(gca,'XScale','log');
axis square;



%% Figure 2b
load('fig2_psa.mat');   
load('fig2_exo.mat');   
load('fig2_endo.mat');  

fig2b = figure; hold on;
title('V1/V2 covert & presaccadic attention');

plot([-0.5,3.6],[-0.5,3.6],'Color',[0,0,0]);
plot([-0.5,3.6],[0,0],'k--');
plot([0,0],[-0.5,3.6],'k--');

scatter(fig2_psa.diff_T_dmax, fig2_psa.diff_n_dmax, 'MarkerFaceColor', colPSA, 'MarkerEdgeColor', [1,1,1]);
scatter(fig2_exo.diff_T_dmax, fig2_exo.diff_n_dmax, 'MarkerFaceColor', colEXO, 'MarkerEdgeColor', [1,1,1]);
scatter(fig2_endo.diff_T_dmax,fig2_endo.diff_n_dmax,'MarkerFaceColor', colENDO,'MarkerEdgeColor', [1,1,1]);

axis square;
set(gca,'Ylim',[-0.5,3.6],'YTick',0:3,'Xlim',[-0.5,3.6],'XTick',0:3);
xlabel('test stimulated (dmax valid-invalid)');
ylabel('test not stimulated (dmax valid-invalid)');



%% Figure 3
load('fig3_V1V2.mat');
load('fig3_FEF.mat');

fig3 = figure;

%V1/V2
subplot(2,3,1); hold on;
title('presac. benefit');
for i = 1:4
    sem = std(fig3_V1V2.valid_T(:,i))./sqrt(size(fig3_V1V2.valid_T,1)-1);
    plot([i,i]-0.1,[mean(fig3_V1V2.valid_T(:,i),1)+sem; mean(fig3_V1V2.valid_T(:,i),1)-sem],'Color',colV);
    sem = std(fig3_V1V2.valid_n(:,i))./sqrt(size(fig3_V1V2.valid_n,1)-1);
    plot([i,i]+0.1,[mean(fig3_V1V2.valid_n(:,i),1)+sem; mean(fig3_V1V2.valid_n(:,i),1)-sem],'Color',colV);
end
plot([1:4]-0.1,mean(fig3_V1V2.valid_T,1),'-o','Color',colV,'MarkerFaceColor',colTMS,'MarkerEdgeColor',colV);
plot([1:4]+0.1,mean(fig3_V1V2.valid_n,1),'-o','Color',colV,'MarkerFaceColor',colSym,'MarkerEdgeColor',colV);
axis square;
set(gca,'Ylim',[0,3.75],'YTick',0:0.75:3.75,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (d'')');

subplot(2,3,2); hold on;
title('presac. cost');
for i = 1:4
    sem = std(fig3_V1V2.invalid_T(:,i))./sqrt(size(fig3_V1V2.invalid_T,1)-1);
    plot([i,i]-0.1,[mean(fig3_V1V2.invalid_T(:,i),1)+sem; mean(fig3_V1V2.invalid_T(:,i),1)-sem],'Color',colI);
    sem = std(fig3_V1V2.invalid_n(:,i))./sqrt(size(fig3_V1V2.invalid_n,1)-1);
    plot([i,i]+0.1,[mean(fig3_V1V2.invalid_n(:,i),1)+sem; mean(fig3_V1V2.invalid_n(:,i),1)-sem],'Color',colI);
end
plot([1:4]-0.1,mean(fig3_V1V2.invalid_T,1),'-o','Color',colI,'MarkerFaceColor',colTMS,'MarkerEdgeColor',colI);
plot([1:4]+0.1,mean(fig3_V1V2.invalid_n,1),'-o','Color',colI,'MarkerFaceColor',colSym,'MarkerEdgeColor',colI);
axis square;
set(gca,'Ylim',[0,3.75],'YTick',0:0.75:3.75,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (d'')');

subplot(2,3,3); hold on;
title('V1/V2 TMS effect');
for i = 1:4
    sem = std(fig3_V1V2.valid_diff(:,i))./sqrt(size(fig3_V1V2.valid_diff,1)-1);
    plot([i,i]-0.1,[mean(fig3_V1V2.valid_diff(:,i),1)+sem; mean(fig3_V1V2.valid_diff(:,i),1)-sem],'Color',colV);
    sem = std(fig3_V1V2.invalid_diff(:,i))./sqrt(size(fig3_V1V2.invalid_diff,1)-1);
    plot([i,i]+0.1,[mean(fig3_V1V2.invalid_diff(:,i),1)+sem; mean(fig3_V1V2.invalid_diff(:,i),1)-sem],'Color',colI);
end
plot([1:4]-0.1,mean(fig3_V1V2.valid_diff,1),'-o','Color',colV,'MarkerFaceColor',colDiff,'MarkerEdgeColor',colV);
plot([1:4]+0.1,mean(fig3_V1V2.invalid_diff,1),'-o','Color',colI,'MarkerFaceColor',colDiff,'MarkerEdgeColor',colI);
axis square;
set(gca,'Ylim',[-1.5,2.25],'YTick',-1.5:0.75:2.25,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (dprime-diff)');


% rFEF+
subplot(2,3,4); hold on;
title('presac. benefit');
for i = 1:4
    sem = std(fig3_FEF.valid_T(:,i))./sqrt(size(fig3_FEF.valid_T,1)-1);
    plot([i,i]-0.1,[mean(fig3_FEF.valid_T(:,i),1)+sem; mean(fig3_FEF.valid_T(:,i),1)-sem],'Color',colV);
    sem = std(fig3_FEF.valid_n(:,i))./sqrt(size(fig3_FEF.valid_n,1)-1);
    plot([i,i]+0.1,[mean(fig3_FEF.valid_n(:,i),1)+sem; mean(fig3_FEF.valid_n(:,i),1)-sem],'Color',colV);
end
plot([1:4]-0.1,mean(fig3_FEF.valid_T,1),'-o','Color',colV,'MarkerFaceColor',colTMS,'MarkerEdgeColor',colV);
plot([1:4]+0.1,mean(fig3_FEF.valid_n,1),'-o','Color',colV,'MarkerFaceColor',colSym,'MarkerEdgeColor',colV);
axis square;
set(gca,'Ylim',[0,3.75],'YTick',0:0.75:3.75,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (d'')');

subplot(2,3,5); hold on;
title('presac. cost');
for i = 1:4
    sem = std(fig3_FEF.invalid_T(:,i))./sqrt(size(fig3_FEF.invalid_T,1)-1);
    plot([i,i]-0.1,[mean(fig3_FEF.invalid_T(:,i),1)+sem; mean(fig3_FEF.invalid_T(:,i),1)-sem],'Color',colI);
    sem = std(fig3_FEF.invalid_n(:,i))./sqrt(size(fig3_FEF.invalid_n,1)-1);
    plot([i,i]+0.1,[mean(fig3_FEF.invalid_n(:,i),1)+sem; mean(fig3_FEF.invalid_n(:,i),1)-sem],'Color',colI);
end
plot([1:4]-0.1,mean(fig3_FEF.invalid_T,1),'-o','Color',colI,'MarkerFaceColor',colTMS,'MarkerEdgeColor',colI);
plot([1:4]+0.1,mean(fig3_FEF.invalid_n,1),'-o','Color',colI,'MarkerFaceColor',colSym,'MarkerEdgeColor',colI);
axis square;
set(gca,'Ylim',[0,3.75],'YTick',0:0.75:3.75,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (d'')');

subplot(2,3,6); hold on;
title('rFEF+ TMS effect');
for i = 1:4
    sem = std(fig3_FEF.valid_diff(:,i))./sqrt(size(fig3_FEF.valid_diff,1)-1);
    plot([i,i]-0.1,[mean(fig3_FEF.valid_diff(:,i),1)+sem; mean(fig3_FEF.valid_diff(:,i),1)-sem],'Color',colV);
    sem = std(fig3_FEF.invalid_diff(:,i))./sqrt(size(fig3_FEF.invalid_diff,1)-1);
    plot([i,i]+0.1,[mean(fig3_FEF.invalid_diff(:,i),1)+sem; mean(fig3_FEF.invalid_diff(:,i),1)-sem],'Color',colI);
end
plot([1:4]-0.1,mean(fig3_FEF.valid_diff,1),'-o','Color',colV,'MarkerFaceColor',colDiff,'MarkerEdgeColor',colV);
plot([1:4]+0.1,mean(fig3_FEF.invalid_diff,1),'-o','Color',colI,'MarkerFaceColor',colDiff,'MarkerEdgeColor',colI);
axis square;
set(gca,'Ylim',[-1.5,2.25],'YTick',-1.5:0.75:2.25,'Xlim',[0,5],'XTick',0.5:4.5,'XTickLabels',0:50:200);
xlabel('TMS rel. to sac onset (ms)');
ylabel('Visual sensitivity (dprime-diff)');



%% Figure S1a
load('fig2_exo.mat');   

figS1a = figure; title('V1/V2 TMS - covert exogenous attention');
hold on;

% fit individual subject's data
n       = size(fig2_exo.valid_T,1);
ncond   = 6;
cLevels = logspace(log10(0.02),log10(0.85),7);
cLevels = cLevels(1,[1,3:end])';

%transform the values to log and transform back after fitting
xx = log10(logspace(log10(0.01), log10(cLevels(end)), 1000)*100); % transform to log the interpolation
logContrast = log10(cLevels*100); % transform to log

ub = [4,logContrast(5),4];      % upper bound on params
lb = [4,logContrast(1),1];      % lower bound on params
x0 = [4,logContrast(3),2];      % starting point

% options for fmincom
opt = optimoptions('fmincon', 'Algorithm', 'interior-point','Display','off', 'MaxIter', 5000);

slope = nan(ncond,n);
c50   = nan(ncond,n);
dmax  = nan(ncond,n);

S = []; fit = []; interp_f = [];
for ii = 1:n % loop through observers

    S = [fig2_exo.valid_T(ii,:)',fig2_exo.neutral_T(ii,:)',fig2_exo.invalid_T(ii,:)',...
         fig2_exo.valid_n(ii,:)',fig2_exo.neutral_n(ii,:)',fig2_exo.invalid_n(ii,:)'];
    
    for jj = 1:size(S,2) % loop through conditions
        
        data = S(:,jj);
        
        f = @(par)nakaRushton(logContrast,par,data);
        [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
        
        [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
        [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit

        % store params
        slope(jj,ii) = par(1);
        c50(jj,ii)   = 10.^par(2); % convert back to contrast 
        dmax(jj,ii)  = par(3);
    end

end

% symbolic group average fit
S = [mean(fig2_exo.valid_T,1)',mean(fig2_exo.neutral_T,1)',mean(fig2_exo.invalid_T,1)',...
     mean(fig2_exo.valid_n,1)',mean(fig2_exo.neutral_n,1)',mean(fig2_exo.invalid_n,1)'];

av_slope = nan(size(S,2),1);
av_c50   = nan(size(S,2),1);
av_dmax  = nan(size(S,2),1);

fit = []; interp_f = [];
for jj = 1:size(S,2) % loop through conditions
    
    data = S(:,jj);
    
    f = @(par)nakaRushton(logContrast,par,data);
    [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
    
    [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
    [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit
    
    % store params
    av_slope(jj,1) = par(1);
    av_c50(jj,1)   = 10.^par(2); % convert back to contrast
    av_dmax(jj,1)  = par(3);
end

antilogC = 10.^(logContrast);
antixx   = 10.^(xx);

mk_size = 7;
line_w  = 2;

condCol_vec    = [colV;colN;colI;colV;colN;colI];
condTMSCol_vec = [colTMS;colTMS;colTMS;colSym;colSym;colSym];


% plot interpolated fits & data points
for i = 1:3
    semilogx(antixx,interp_f(:,i),'-','color',condCol_vec(i,:),'linewidth',line_w);
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end
for i = 4:6
    semilogx(antixx,interp_f(:,i),'-.','color',condCol_vec(i,:),'linewidth',line_w); hold on;
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end 

% dmax
for i = 1:3
    plot([145,145],[mean(dmax(i,:),2) - std(dmax(i,:))./sqrt(n-1), mean(dmax(i,:),2) + std(dmax(i,:))./sqrt(n-1)],'Color',condCol_vec(i,:),'linewidth',line_w);
    plot([120,120],[mean(dmax(i+3,:),2) - std(dmax(i+3,:))./sqrt(n-1), mean(dmax(i+3,:),2) + std(dmax(i+3,:))./sqrt(n-1)],'Color',condCol_vec(i+3,:),'linewidth',line_w);
    
    scatter([145],mean(dmax(i,:),2),80,'MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:));
    scatter([120],mean(dmax(i+3,:),2),80,'MarkerEdgeColor',condCol_vec(i+3,:),'MarkerFaceColor',condTMSCol_vec(i+3,:));
end

set(gca,'Ylim',[0 3.75],'YTick',0:3,'Xlim',[0 150],'XTick',round(cLevels*100)');
xlabel('contrast (%)'); ylabel('visual sensitivity (d'')');
set(gca,'XScale','log');
axis square;



%% Figure S1b
load('fig2_endo.mat');   

figS1b = figure; title('V1/V2 TMS - covert endogenous attention');
hold on;

% fit individual subject's data
n       = size(fig2_endo.valid_T,1);
ncond   = 6;
cLevels = logspace(log10(0.02),log10(0.85),8)';

%transform the values to log and transform back after fitting
xx = log10(logspace(log10(0.01), log10(cLevels(end)), 1000)*100); % transform to log the interpolation
logContrast = log10(cLevels*100); % transform to log

ub = [4,logContrast(end),5];	% upper bound on params
lb = [4,logContrast(1),1];      % lower bound on params
x0 = [4,logContrast(3),2];   	% starting point

% options for fmincom
opt = optimoptions('fmincon', 'Algorithm', 'interior-point','Display','off', 'MaxIter', 5000);

slope = nan(ncond,n);
c50   = nan(ncond,n);
dmax  = nan(ncond,n);

S = []; fit = []; interp_f = [];
for ii = 1:n % loop through observers

    S = [fig2_endo.valid_T(ii,:)',fig2_endo.neutral_T(ii,:)',fig2_endo.invalid_T(ii,:)',...
         fig2_endo.valid_n(ii,:)',fig2_endo.neutral_n(ii,:)',fig2_endo.invalid_n(ii,:)'];
    
    for jj = 1:size(S,2) % loop through conditions
        
        data = S(:,jj);
        
        f = @(par)nakaRushton(logContrast,par,data);
        [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
        
        [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
        [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit

        % store params
        slope(jj,ii) = par(1);
        c50(jj,ii)   = 10.^par(2); % convert back to contrast 
        dmax(jj,ii)  = par(3);
    end

end

% symbolic group average fit
S = [mean(fig2_endo.valid_T,1)',mean(fig2_endo.neutral_T,1)',mean(fig2_endo.invalid_T,1)',...
     mean(fig2_endo.valid_n,1)',mean(fig2_endo.neutral_n,1)',mean(fig2_endo.invalid_n,1)'];

av_slope = nan(size(S,2),1);
av_c50   = nan(size(S,2),1);
av_dmax  = nan(size(S,2),1);

fit = []; interp_f = [];
for jj = 1:size(S,2) % loop through conditions
    
    data = S(:,jj);
    
    f = @(par)nakaRushton(logContrast,par,data);
    [par,fval,exitflag,output,grad,hessian] = fmincon(f,x0,[],[],[],[],lb,ub,[],opt);
    
    [~, fit(:,jj)] = nakaRushton(logContrast,par);  % actual fit
    [~, interp_f(:,jj)] = nakaRushton(xx,par);      % interpolate fit
    
    % store params
    av_slope(jj,1) = par(1);
    av_c50(jj,1)   = 10.^par(2); % convert back to contrast
    av_dmax(jj,1)  = par(3);
end

antilogC = 10.^(logContrast);
antixx   = 10.^(xx);

mk_size = 7;
line_w  = 2;

condCol_vec    = [colV;colN;colI;colV;colN;colI];
condTMSCol_vec = [colTMS;colTMS;colTMS;colSym;colSym;colSym];


% plot interpolated fits & data points
for i = 1:3
    semilogx(antixx,interp_f(:,i),'-','color',condCol_vec(i,:),'linewidth',line_w);
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end
for i = 4:6
    semilogx(antixx,interp_f(:,i),'-.','color',condCol_vec(i,:),'linewidth',line_w); hold on;
    semilogx(antilogC,S(:,i),'o','MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:),'Markersize',mk_size); hold on;
end 

% dmax
for i = 1:3
    plot([145,145],[mean(dmax(i,:),2) - std(dmax(i,:))./sqrt(n-1), mean(dmax(i,:),2) + std(dmax(i,:))./sqrt(n-1)],'Color',condCol_vec(i,:),'linewidth',line_w);
    plot([120,120],[mean(dmax(i+3,:),2) - std(dmax(i+3,:))./sqrt(n-1), mean(dmax(i+3,:),2) + std(dmax(i+3,:))./sqrt(n-1)],'Color',condCol_vec(i+3,:),'linewidth',line_w);
    
    scatter([145],mean(dmax(i,:),2),80,'MarkerEdgeColor',condCol_vec(i,:),'MarkerFaceColor',condTMSCol_vec(i,:));
    scatter([120],mean(dmax(i+3,:),2),80,'MarkerEdgeColor',condCol_vec(i+3,:),'MarkerFaceColor',condTMSCol_vec(i+3,:));
end

set(gca,'Ylim',[0 3.75],'YTick',0:3,'Xlim',[0 150],'XTick',round(cLevels*100)');
xlabel('contrast (%)'); ylabel('visual sensitivity (d'')');
set(gca,'XScale','log');
axis square;


%% Figure S2
load('figS2_err.mat'); 
load('figS2_lat.mat'); 

figS2 = figure;

subplot(2,2,1); hold on;
title('V1/V2 TMS');
for i = 1:5
    sem = std(figS2_lat.V1V2_stim(:,i))./sqrt(size(figS2_lat.V1V2_stim,1)-1);
    plot([i,i]-0.1,[mean(figS2_lat.V1V2_stim(:,i),1)+sem; mean(figS2_lat.V1V2_stim(:,i),1)-sem],'Color',colStim);
    sem = std(figS2_lat.V1V2_not(:,i))./sqrt(size(figS2_lat.V1V2_not,1)-1);
    plot([i,i]+0.1,[mean(figS2_lat.V1V2_not(:,i),1)+sem; mean(figS2_lat.V1V2_not(:,i),1)-sem],'Color',colNot);
end
plot([1:5]-0.1,mean(figS2_lat.V1V2_stim,1),'Color',colStim);
plot([1:5]+0.1,mean(figS2_lat.V1V2_not,1),'Color',colNot);

axis square;
set(gca,'Ylim',[220,260],'Xlim',[0,6],'XTick',1:5,'XTickLabels',0:50:200,'YTick',220:10:260);
xlabel('TMS rel. to cue onset (ms)');
ylabel('Saccade latency (ms)');


subplot(2,2,2); hold on;
title('V1/V2 TMS');
for i = 1:5
    sem = std(figS2_err.V1V2_stim(:,i))./sqrt(size(figS2_err.V1V2_stim,1)-1);
    plot([i,i]-0.1,[mean(figS2_err.V1V2_stim(:,i),1)+sem; mean(figS2_err.V1V2_stim(:,i),1)-sem],'Color',colStim);
    sem = std(figS2_err.V1V2_not(:,i))./sqrt(size(figS2_err.V1V2_not,1)-1);
    plot([i,i]+0.1,[mean(figS2_err.V1V2_not(:,i),1)+sem; mean(figS2_err.V1V2_not(:,i),1)-sem],'Color',colNot);
end
plot([1:5]-0.1,mean(figS2_err.V1V2_stim,1),'Color',colStim);
plot([1:5]+0.1,mean(figS2_err.V1V2_not,1),'Color',colNot);

axis square;
set(gca,'Ylim',[0.5,1.5],'Xlim',[0,6],'XTick',0.5:5.5,'XTickLabels',-250:50:0,'YTick',0.5:0.25:1.5);
xlabel('TMS rel. to saccade onset (ms)');
ylabel('Saccade landing error (deg)');


subplot(2,2,3); hold on;
title('rFEF+ TMS');
for i = 1:5
    sem = std(figS2_lat.FEF_stim(:,i))./sqrt(size(figS2_lat.FEF_stim,1)-1);
    plot([i,i]-0.1,[mean(figS2_lat.FEF_stim(:,i),1)+sem; mean(figS2_lat.FEF_stim(:,i),1)-sem],'Color',colStim);
    sem = std(figS2_lat.FEF_not(:,i))./sqrt(size(figS2_lat.FEF_not,1)-1);
    plot([i,i]+0.1,[mean(figS2_lat.FEF_not(:,i),1)+sem; mean(figS2_lat.FEF_not(:,i),1)-sem],'Color',colNot);
end
plot([1:5]-0.1,mean(figS2_lat.FEF_stim,1),'Color',colStim);
plot([1:5]+0.1,mean(figS2_lat.FEF_not,1),'Color',colNot);

axis square;
set(gca,'Ylim',[220,260],'Xlim',[0,6],'XTick',1:5,'XTickLabels',0:50:200,'YTick',220:10:260);
xlabel('TMS rel. to cue onset (ms)');
ylabel('Saccade latency (ms)');


subplot(2,2,4); hold on;
title('rFEF+ TMS');
for i = 1:5
    sem = std(figS2_err.FEF_stim(:,i))./sqrt(size(figS2_err.FEF_stim,1)-1);
    plot([i,i]-0.1,[mean(figS2_err.FEF_stim(:,i),1)+sem; mean(figS2_err.FEF_stim(:,i),1)-sem],'Color',colStim);
    sem = std(figS2_err.FEF_not(:,i))./sqrt(size(figS2_err.FEF_not,1)-1);
    plot([i,i]+0.1,[mean(figS2_err.FEF_not(:,i),1)+sem; mean(figS2_err.FEF_not(:,i),1)-sem],'Color',colNot);
end
plot([1:5]-0.1,mean(figS2_err.FEF_stim,1),'Color',colStim);
plot([1:5]+0.1,mean(figS2_err.FEF_not,1),'Color',colNot);

axis square;
set(gca,'Ylim',[0.5,1.5],'Xlim',[0,6],'XTick',0.5:5.5,'XTickLabels',-250:50:0,'YTick',0.5:0.25:1.5);
xlabel('TMS rel. to saccade onset (ms)');
ylabel('Saccade landing error (deg)');

