%student: Rita Leite
%matlab code:

clear
close all

data = load("pendulumData.mat");

n_threads = 128;

rng(23);

x1_GT = data.x1;
x2_GT = data.x2;
y = data.y;

g = 9.81;
q_c = 0.05;
s2 = 0.1;
dt = 0.01;
N = length(y);

t = 0:dt:(N - 1)*dt;

Q = q_c * [ dt^3 / 3, dt^2/2; dt^2/2, dt];

% We are able to sample N( 0, Q) with L * w, where w ~ randn( 2, 1 )
L = chol( Q );


J = 10000;

% Let's sample state from a prior distribution
x1_Prior = 0.1 * randn( J, 1);
x2_Prior = 1.2 + 0.1 * randn( J, 1);


X1 = zeros( J, N);
X2 = zeros( J, N);

ll_host=single(zeros(J ,1));
tic();
%Kernel Initialization
k_x = parallel.gpu.CUDAKernel('./pendulum.ptx', './pendulum.cu','pendulum_propagate');
k_x.GridSize = [ceil((J*2+n_threads+1)/n_threads) 1];
k_x.ThreadBlockSize = [n_threads 1 1];   
k_log = parallel.gpu.CUDAKernel('./log.ptx', './log.cu','log_normal');
k_log.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_log.ThreadBlockSize = [n_threads 1];        
k_sum = parallel.gpu.CUDAKernel('./log.ptx', './log.cu','sum_sequential');
k_sum.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_sum.ThreadBlockSize = [n_threads 1];
k_sum.SharedMemorySize = n_threads * 4;  
k_max = parallel.gpu.CUDAKernel('./log.ptx', './log.cu','get_max');
k_max.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_max.ThreadBlockSize = [n_threads 1];
k_max.SharedMemorySize = n_threads * 4; 
k_normal=parallel.gpu.CUDAKernel('./log.ptx', './log.cu','normalise_ws');
k_normal.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_normal.ThreadBlockSize = [n_threads 1];
k_un = parallel.gpu.CUDAKernel('./log.ptx', './log.cu','unNormalised_ws');
k_un.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_un.ThreadBlockSize = [n_threads 1];

k_sum2 = parallel.gpu.CUDAKernel('./log.ptx', './log.cu','sum_sequential');
k_sum2.GridSize = [ceil((J+n_threads-1)/n_threads) 1];
k_sum2.ThreadBlockSize = [n_threads 1];
k_sum2.SharedMemorySize = n_threads * 4; 

   
        




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
    xNext = zeros( J, 2);
    eps1 = gpuArray(randn([1 J],'single'));
    eps2 = gpuArray(randn([1 J],'single'));

    %------ Propagation step -----
          
    
    %Pass variables to GPU
    xNext_device1 = gpuArray(single(xNext( :, 1)));
    xNext_device2 = gpuArray(single(xNext( :, 2)));

    L_device = gpuArray(single(L));
    
   
   
    xPrev_device1 = gpuArray(single(x1_Previous));
    xPrev_device2 = gpuArray(single(x2_Previous));
	
    %Execute kernel and bring results to host
    [xNext_device1 xNext_device2] = feval(k_x, xPrev_device1,xPrev_device2, dt, L_device, eps1,eps2, xNext_device1,xNext_device2, J);
    xNext( :,1) = gather(xNext_device1);	
    xNext( :,2) = gather(xNext_device2);	
    
       
    
    %Propagation Step Done; Now we get the weights
    
    
    %Calculate loglikelihood for each value 
    
    ll_device = gpuArray(ll_host);
    ll_device = feval(k_log, xNext_device1,0,s2,ll_device,J,y_n);
    
    %Calculate Maximium of log likelihood
        
    host = zeros([k_max.GridSize(1) 1], 'single');
    device = gpuArray(host); % allocate memory for the second array
    max1 = feval(k_max, ll_device, device, J);
    
    
        
    max_w=max(gather(max1));
        

    %Get unNormalized weights
    
    %Get W
    sum_all=zeros([k_max.GridSize(1) 1], 'single');
    sum_all_device=gpuArray(sum_all);

    W_kk= gpuArray(zeros([J 1],'single'));
    W_kk = feval(k_un,ll_device, max_w, W_kk,J);

    
    %Get sum of W_i
    %counter = ceil((J+n_threads-1)/n_threads);
    sum_all_device = feval(k_sum2, sum_all_device, J , W_kk);
    
    sum_all=sum(gather(sum_all_device));
    
    

    %Normalize the weights

    W_kk = feval(k_normal,W_kk,sum_all,J);
    ws_CUDA = gather(W_kk);
    
    %Resampling
    resampleInds = resampleResidual( ws_CUDA );
    xNext = xNext( resampleInds, :);
    
    X1( :, ii) = xNext( :, 1);
    X2( :, ii) = xNext( :, 2);

end
toc;

Y = sin( X1 );

%%Plots
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

saveas(f,'plots2.png');









function [ll] = modelLogLikelihood( x, y, s2)

    N = length( x );
    
    % theta_1 + theta_2 * x
    residuals = y - sin( x);
    ll = -0.5 * N * log( 2 * pi ) - 0.5 * N * log( s2 );
    ll = ll - residuals' * residuals / ( 2 * s2 );
end

function [ indx ] = resampleResidual_2( w , n_threads)

M = length(w);
k_dotprod= parallel.gpu.CUDAKernel('./resample.ptx', './resample.cu','matrix_const_product');
k_dotprod.GridSize = [ceil((M+n_threads-1)/n_threads) 1];
k_dotprod.ThreadBlockSize = [n_threads 1];
k_dotprod.SharedMemorySize = n_threads * 4;  



% "Repetition counts" (plus the random part, later on):
Ns = zeros([M 1],'single');
% The "remainder" or "residual" count:
%res_sum = zeros([k_aux.GridSize(1) 1], 'single');
%res_sum_device=gpuArray(res_sum);
Ns_device=gpuArray(Ns);

Ns_device = feval(k_dotprod,  Ns_device , w, M);
R=sum(gather(Ns_device));
Ns=gather(Ns_device);
% The number of particles which will be drawn stocastically:
M_rdn = M-R;
% The modified weights:
Ws = (M .* w - Ns)/M_rdn;
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