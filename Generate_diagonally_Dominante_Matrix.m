% This is a simple code to generate a diagonally dominant matrix A
% It implement Neumann series approximations of orders 2, 3 and 4
clc
clear all
close all
% Matrix dimention 64x64
A_dim=64;
H_dim=16;

% Generating a tall matrix H such that A=H'*H is diagonally dominant matrix
H=randn(H_dim*A_dim,A_dim);
% A=ceil(H'*H/(H_dim*A_dim)*2^(0));
A=(H'*H/(H_dim*A_dim)*2^(0));
% Save the matrix A to a text file
writematrix(A, 'MatrixA.txt');

% Direct (exact) matrix inversion
Ainv=inv(A);
% Isolate diagonal and off-diagonal parts of A
D=diag(diag(A));
E=A-D;
Dinv=inv(D);
temp=(Dinv*E)
temp2=temp*Dinv
mult3 = (temp*temp)*Dinv
Approx = Dinv-temp2+ mult3
% Computing Neumann series approximation of order 2, 3 and 4
Ainv2=Dinv-Dinv*E*Dinv;
Ainv3=Dinv-Dinv*E*Dinv+Dinv*E*Dinv*E*Dinv;
Ainv4=Dinv-Dinv*E*Dinv+Dinv*E*Dinv*E*Dinv-Dinv*E*Dinv*E*Dinv*E*Dinv;
% Approximate inversion errors for comparaison
Error2=norm(Ainv/norm(Ainv)-Ainv2/norm(Ainv2),'fro');
Error3=norm(Ainv/norm(Ainv)-Ainv3/norm(Ainv3),'fro');
Error4=norm(Ainv/norm(Ainv)-Ainv4/norm(Ainv4),'fro');