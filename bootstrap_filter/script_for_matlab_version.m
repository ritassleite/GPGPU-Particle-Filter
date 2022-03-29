
%clear
%close all

data = load("pendulumData.mat");
rng(23);
x1_GT = data.x1;
x2_GT = data.x2;
y = data.y;

g = 9.81;
q_c = 0.05;
s2 = 0.1;
dt = 0.01;
N = length( y );

t = 0:dt:(N - 1)*dt;

Q = q_c * [ dt^3 / 3, dt^2/2; dt^2/2, dt];

% We are able to sample N( 0, Q) with L * w, where w ~ randn( 2, 1 )
L = chol( Q );

N = length( y );
J = 10000;

% Let's sample state from a prior distribution
x1_Prior = 0.1 * randn( J, 1);
x2_Prior = 1.2 + 0.1 * randn( J, 1);

X1 = zeros( J, N);
X2 = zeros( J, N);

tic
for ii = 1:N
    
    if( ii == 1 )
        x1_Previous = x1_Prior;
        x2_Previous = x2_Prior;
    else
        x1_Previous = X1( :, ii - 1);
        x2_Previous = X2( :, ii - 1);
    end
    
    % Get current data point for likelihood computation
    y_n = y(ii);
    ll = zeros( J, 1);
    xNext = zeros( J, 2);
    
    % Propagation step
    for jj = 1:J
        
        x1_jj = x1_Previous(jj);
        x2_jj = x2_Previous(jj);
        
        % Sample the random perturbation of the state
        w = randn( 2, 1);
        q = L * w;

        % Propagate
        x1_Next_jj = x1_jj + x2_jj * dt;
        x2_Next_jj = x2_jj - g * sin( x1_jj ) * dt;
        xNext_jj = [ x1_Next_jj; x2_Next_jj] + q;
        
        % Bookkeeping of the particles
        xNext(jj,:) = xNext_jj;
        ll(jj) = modelLogLikelihood( x1_Next_jj, y_n, s2);
    end
    
    if(ii==1)
	next_1=xNext(:,1);
	ll_toc=ll;
    end

    % Compute weights
    ll = ll - max( ll );

    weights = exp( ll );
    weights = weights / sum( weights );
    
    % Resampling
    resampleInds = resampleResidual( weights );
    xNext = xNext( resampleInds, :);
    
    X1( :, ii) = xNext( :, 1);
    X2( :, ii) = xNext( :, 2);
end

Y = sin( X1 );

toc
%%
close all

figure();
f=tiledlayout('flow');

p = [ 0.025, 0.50, 0.975];
X1_CI = quantile( X1, p, 1);

X1_lowerBound = X1_CI( 1, :);
X1_mean = X1_CI( 2, :);
X1_upperBound = X1_CI( 3, :);

tArea = [ t, fliplr(t)];
X1_Area = [ X1_lowerBound, fliplr( X1_upperBound )];

X2_CI = quantile( X2, p, 1);

X2_lowerBound = X2_CI( 1, :);
X2_mean = X2_CI( 2, :);
X2_upperBound = X2_CI( 3, :);

X2_Area = [ X2_lowerBound, fliplr( X2_upperBound )];

Y_CI = quantile( Y, p, 1);

Y_lowerBound = Y_CI( 1, :);
Y_mean = Y_CI( 2, :);
Y_upperBound = Y_CI( 3, :);

Y_Area = [ Y_lowerBound, fliplr( Y_upperBound )];

nexttile();
hold on

h = plot( t, x1_GT, '.');
h = plot( t, X1_mean );
h.LineWidth = 2;

color = h.Color;

hF = fill( tArea, X1_Area, color);
hF.FaceAlpha = 0.35;
hF.LineStyle = 'none';

h = title('$ x_1 $');
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 15;

nexttile();
hold on

h = plot( t, x2_GT, '.');
h = plot( t, X2_mean );
h.LineWidth = 2;

color = h.Color;

hF = fill( tArea, X2_Area, color);
hF.FaceAlpha = 0.35;
hF.LineStyle = 'none';

h = title('$ x_2 $');
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 15;

nexttile();
hold on

h = plot( t, y, '.');
h = plot( t, Y_mean );
h.LineWidth = 2;

color = h.Color;

hF = fill( tArea, Y_Area, color);
hF.FaceAlpha = 0.35;
hF.LineStyle = 'none';

h = title("$y$");
h.Interpreter = "latex";

ax = gca();
ax.FontSize = 15;

saveas(f,'plots.png');

function [ll] = modelLogLikelihood( alpha, y_n, s2)

    N = 1;
    
    % theta_1 + theta_2 * x
    residuals = y_n - sin( alpha );
    ll = -0.5 * N * log( 2 * pi ) - 0.5 * N * log( s2 );
    ll = ll - residuals' * residuals / ( 2 * s2 );
    
end

function [ indx ] = resampleResidual( w )
M = length(w);
% "Repetition counts" (plus the random part, later on):
Ns = floor(M .* w);
% The "remainder" or "residual" count:
R = sum( Ns );
% The number of particles which will be drawn stocastically:
M_rdn = M-R;
% The modified weights:
Ws = (M .* w - floor(M .* w))/M_rdn;
% Draw the deterministic part:
% ---------------------------------------------------
i=1;
for j=1:M,
    for k=1:Ns(j),
        indx(i)=j;
        i = i +1;
    end
end;
% And now draw the stocastic (Multinomial) part:
% ---------------------------------------------------
Q = cumsum(Ws);
Q(M)=1; % Just in case...
while (i<=M),
    sampl = rand(1,1);  % (0,1]
    j=1;
    while (Q(j)<sampl),
        j=j+1;
    end;
    indx(i)=j;
    i=i+1;
end
end