function phi = cvx_qsp(func, deg, opts)
%% Find the phase factors associate with a given smooth function.
%
% cvx_qsp first uses cvx to find a polynomial approximating the function
% on one or two intervals on [0,1]. Then it uses qsppack to evaluate the
% phase factors. It returns phi_qc that can be readily plug into
% a QSVT circuit.
%
% Lin Lin
% Last revision: 11/06/2021

if( ~isfield(opts, 'npts') )
  opts.npts = 200;
end
if( ~isfield(opts, 'epsil') )
  opts.epsil = 0.001;
end
if( ~isfield(opts, 'fscale') )
  opts.fscale = 1-opts.epsil;
end
if( ~isfield(opts, 'intervals') )
  opts.intervals = [0,1];
end
if( ~isfield(opts, 'criteria') )
  opts.criteria = 1e-10;
end
if( ~isfield(opts, 'isplot') )
  opts.isplot = true;
end
if( ~isfield(opts, 'fname') )
  opts.fname = 'func';
end
if( ~isfield(opts, 'objnorm') )
  opts.objnorm = inf;
end

% check variables and assign local variables
assert(mod(length(opts.intervals), 2) == 0);
parity = mod(deg, 2);
epsil = opts.epsil;
npts = opts.npts;

[xpts, wpts] = chebpts(2*npts);
ind = find(xpts>0);
xpts = xpts(ind);
wpts = wpts(ind)';

n_interval = length(opts.intervals) / 2;
ind_union = [];
for i = 1 : n_interval
  intervals{i} = find((xpts >= opts.intervals(2*i-1)) &...
    (xpts<=opts.intervals(2*i)));
  ind_union = union(ind_union, intervals{i});
end


% evaluate the target function
fx = zeros(npts,1);
fx(ind_union) = opts.fscale * func(xpts(ind_union));


% prepare the Chebyshev polynomials
if( parity == 0 )
  n_coef = deg / 2 + 1;
else
  n_coef = (deg + 1) / 2;
end
Ax = zeros(npts, n_coef);
for k = 1 : n_coef
  if( parity == 0 )
    Tcheb = chebpoly(2*(k-1));
  else
    Tcheb = chebpoly(2*k-1);
  end
  Ax(:,k) = Tcheb(xpts);
end


%% Use CVX to optimize the Chebyshev coefficients
cvx_begin quiet
variable coef(n_coef)
variable y(npts)
minimize ( norm(y(ind_union)-fx(ind_union), opts.objnorm)) 

y==Ax*coef
y>=-(1-epsil)
y<=(1-epsil)
cvx_end

fprintf('norm error = %g\n', norm(y(ind_union)-fx(ind_union),inf));


% use chebfun to make sure the maximum is less than 1
coef_full = zeros(deg+1,1);
if( parity == 0 )
  coef_full(1:2:end) = coef;
else
  coef_full(2:2:end) = coef;
end
sol_cheb = chebfun(coef_full, 'coeffs');
max_sol = max(sol_cheb);
fprintf('max of solution = %g\n', max_sol);
if( max_sol > 1.0-1e-10 )
  error('Solution is not bounded by 1. Increase npts');
end

%% Compute the QSP phase factors
opts_qsp.criteria = opts.criteria;
[phi,out] = QSP_solver(coef, parity,opts_qsp);

end
