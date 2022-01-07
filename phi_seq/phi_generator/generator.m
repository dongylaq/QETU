%% QSVT circuit with Hadamard block encoding for spin problems.
%
% H = -(Z_1 * Z_2 + X_1 + X_2)
% 
% This routine uses the arccos transformation, which is more natural
% when combined with the square root access to compute the energy. 
% 
% This uses the further simplified qsvt circuit without the ancilla
% qubit or the explicit Hadamard block encoding.
%
% This uses the covention in the paper, and also outputs the QSP phase
% factors that can be used in the hadamardbe code.
%
% Lin Lin
% Last revision: 1/5/2021

% nsys = 1;
nsys = 2;
% nsys = 4;

% the subscript s means single qubit 
Z_s = [1 0; 0 -1];
I_s = eye(2);
X_s = [0 1; 1 0];
H_s = 1/sqrt(2) * [1 1; 1 -1];
S_s = [1 0; 0 1i];
p0_s = [1 0; 0 0];
p1_s = [0 0; 0 1];

sigma_to_lambda = @(x) acos(2*x.^2-1);
lambda_to_sigma = @(x) sqrt((1+cos(x))./2);

X = cell(nsys,1);
Z = cell(nsys,1);
for i = 1 : nsys
  Z{i} = kron(eye(2^(i-1)), kron(Z_s, eye(2^(nsys-i)))); 
  X{i} = kron(eye(2^(i-1)), kron(X_s, eye(2^(nsys-i)))); 
end
N_sys = 2^nsys;
I_sys = eye(N_sys);

if( nsys == 1 )
  % one qubit field model
  H1 = -1/sqrt(2) * Z{1};
  H2 = -1/sqrt(2) * X{1};
end
if( nsys == 2 )
  H1 = -Z{1} * Z{2};
  H2 = -(X{1} + X{2});
end
if( nsys == 4 )
  H1 = -Z{1} * Z{2} - Z{2}*Z{3} - Z{3}*Z{4};
  H2 = -(X{1} + X{2} + X{3} + X{4});
end


% separate form for Trotter
H = H1 + H2;
[ew_H, ev_H] = eig(H);
ev_H = diag(ev_H);
E_gs = ev_H(1);

psi_gs = ew_H(:,1);
psi0 = zeros(2^nsys,1);
psi0(1) = 1.0;

ovlp = abs(psi0'*psi_gs);
fprintf('ovlp = %g\n', ovlp);

% distance between the spectrum and 0, pi.
dist = 0.1;
% shift and scale H
c1 = (pi-2*dist) / (ev_H(end) - ev_H(1));
c2 = dist - c1 * ev_H(1);
% linearly transformed H
H_lt = H * c1 + c2 * eye(N_sys); 
ev_H_lt = ev_H * c1 + c2;
H1_lt = H1 * c1 + c2 * eye(N_sys);
H2_lt = H2 * c1;
dt = c1;


% method = 'trotter2';
method = 'exact';

U_exact = expm(-1i*H_lt);
if( strcmp(method, 'exact') )
  U = U_exact;
end
if( strcmp(method, 'trotter2') )
  % second order Trotter method
  n_tau = 3;
  tau = 1.0 / n_tau;
  U_tau = expm(-1i*H2_lt/2)*expm(-1i*H1_lt)*expm(-1i*H2_lt/2);
  U = U_tau^n_tau;
  fprintf('Trotter error = %g\n', norm(U-U_exact,2));
end

cU = kron(p0_s, I_sys) + kron(p1_s, U);
% Hadamard block encoding. Not explicitly used anymore.
HadBE = kron(H_s, I_sys) * cU * kron(H_s, I_sys);

% check the relation between the singular values and eigenvalues
A = HadBE(1:N_sys, 1:N_sys);
sigma = svd(A);

ev_recon = sort(sigma_to_lambda(sigma));

fprintf('error of reconstructed eigenvalues from Hadamard BE = %g\n',...
  norm(ev_recon-ev_H_lt));

mu = 0.5 * (ev_recon(1) + ev_recon(2));
gap = (ev_recon(2) - ev_recon(1));
E_min = dist/2;
E_max = pi-dist;

sigma_min = lambda_to_sigma(E_max);
sigma_mu = lambda_to_sigma(mu);
sigma_mu_m = lambda_to_sigma(mu+gap/2);
sigma_mu_p = lambda_to_sigma(mu-gap/2);
sigma_max = lambda_to_sigma(E_min);

func_filter = @(x) (x>sigma_mu) * 1.0;

opts.npts = 400;
opts.epsil = 0.01;
opts.fscale = 0.9;
opts.criteria=1e-6;
opts.intervals= [sigma_min, sigma_mu_m, sigma_mu_p, sigma_max];
opts.isplot = true;
opts.fname = 'func';
opts.objnorm = inf;
deg = 20;

phi_qc = cvx_qsp(func_filter, deg, opts);

% output the phase factors
% fname = sprintf('phi_gsp_n%d_d%d.txt', nsys, deg);
% writematrix(phi_qc, fname)

dlmwrite("../ground_state/phi_gsp_n"+int2str(nsys)+"_d"+int2str(deg)+".txt",...
 [phi_qc;c1;c2;opts.fscale], 'precision','%25.15f')