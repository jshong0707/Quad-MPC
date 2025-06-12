import casadi as ca
import numpy as np

# Define dimensions
nx = 4  # 상태 변수 차원
nu = 2  # 제어 입력 차원
N = 10  # 예측 지평선 길이

# Create the CasADi Opti problem
opti = ca.Opti()

# Define symbolic variables for control input
U = opti.variable(nu, N)  # 제어 입력 시퀀스 (N) 크기

# Define reference trajectory (주어진 목표 상태)
x_ref = np.array([1.0, 0.0, 0.0, 0.0])  # 예시로 상태 목표값 설정

# Define state transition matrix (A, B, and g from the discretized model)
A = np.eye(nx)  # 예시: 단위 행렬로 설정
B = np.eye(nx, nu)  # 예시: 단위 행렬로 설정
g = np.zeros(nx)  # 예시: 0 벡터로 설정

# Initial state (x0) and reference trajectory (Xr)
x0 = np.zeros(nx)  # 초기 상태
Xr = np.tile(x_ref, (N+1, 1)).T  # reference trajectory

# Cost weights
Q = np.eye(nx) * 10.0   # 상태에 대한 비용 가중치
R = np.eye(nu) * 1.0    # 제어 입력에 대한 비용 가중치
Qf = np.eye(nx) * 20.0  # 마지막 상태에 대한 비용 가중치

# Define the state trajectory (X) based on the dynamics
X = [x0]  # 초기 상태로 시작
for k in range(N):
    next_state = ca.mtimes([A, X[-1]]) + ca.mtimes([B, U[:, k]]) + g  # 상태 계산
    X.append(next_state)

# Convert list to a CasADi MX array for matrix operations
X = ca.vertcat(*X)  # X를 CasADi MX 배열로 합침


# Define cost function
cost = 0
for k in range(N):
    dx = X[4*k:4*(k+1)] - x_ref  # 상태 오차 (4차원 상태 벡터)
    cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])  # 상태 오차와 제어 입력에 대한 비용

# 마지막 상태에 대한 비용
dxN = X[4*N:4*(N+1)] - x_ref  # 마지막 상태 오차
cost += ca.mtimes([dxN.T, Qf, dxN])  # 마지막 상태에 대한 오차

# Set the objective function
opti.minimize(cost)

# Set some constraints (예시: 제어 입력이 제한 범위 내에 있어야 한다)
opti.subject_to(opti.bounded(-10, U, 10))  # 제어 입력의 범위

# Set the solver
opti.solver('ipopt', {
    'print_time': False,
    'print_in': False,  # 입력값 출력 끄기
    'print_out': False  # 출력값 출력 끄기
})

# Set initial guess for control inputs
opti.set_initial(U, np.zeros((nu, N)))    # 초기 제어 입력

# Solve the optimization problem
sol = opti.solve()

# Extract the solution for control input
U_opt = sol.value(U)  # 최적 제어 입력

print("Optimal control trajectory:", U_opt)
