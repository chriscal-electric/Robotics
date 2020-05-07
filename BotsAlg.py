import numpy as np
import sympy as sp


mt = sp.symbols('vac')


def rad_2_deg(angle):
    return angle * 180/sp.pi

def deg_2_rad(angle):
    return angle * sp.pi/180

def Rx(theta_deg):
    theta_rad = deg_2_rad(theta_deg)
    R = np.array(
        [
            [1, 0, 0],
            [0, sp.cos(theta_rad), -sp.sin(theta_rad)],
            [0, sp.sin(theta_rad), sp.cos(theta_rad)]
        ]
    )
    return R

def Ry(theta_deg):
    theta_rad = deg_2_rad(theta_deg)
    R = np.array(
        [
            [sp.cos(theta_rad), 0, sp.sin(theta_rad)],
            [0, 1, 0],
            [-sp.sin(theta_rad), 0, sp.cos(theta_rad)]
        ]
    )
    return R

def Rz(theta_deg):
    theta_rad = deg_2_rad(theta_deg)
    R =  np.array(
        [
            [sp.cos(theta_rad), -sp.sin(theta_rad), 0],
            [sp.sin(theta_rad), sp.cos(theta_rad), 0],
            [0, 0, 1]
        ]
    )
    return R

def Rxrad(theta_rad):
    R = np.array(
        [
            [1, 0, 0],
            [0, sp.cos(theta_rad), -sp.sin(theta_rad)],
            [0, sp.sin(theta_rad), sp.cos(theta_rad)]
        ]
    )
    return R

def Ryrad(theta_rad):
    R = np.array(
        [
            [sp.cos(theta_rad), 0, sp.sin(theta_rad)],
            [0, 1, 0],
            [-sp.sin(theta_rad), 0, sp.cos(theta_rad)]
        ]
    )
    return R

def Rzrad(theta_rad):
    R =  np.array(
        [
            [sp.cos(theta_rad), -sp.sin(theta_rad), 0],
            [sp.sin(theta_rad), sp.cos(theta_rad), 0],
            [0, 0, 1]
        ]
    )
    return R

def Rotation(Angle_of_rotation, Axis_of_rotation):
    bx = False
    by = False
    bz = False
    if(Axis_of_rotation == 'x' or Axis_of_rotation == 'X'):
        bx = True
        R = Rx(Angle_of_rotation)
    if(Axis_of_rotation == 'y' or Axis_of_rotation == 'Y'):
        by = True
        R = Ry(Angle_of_rotation)
    if(Axis_of_rotation == 'z' or Axis_of_rotation == 'Z'):
        bz = True
        R = Rz(Angle_of_rotation)

    return R

def Rotationrad(Angle_of_rotation, Axis_of_rotation):
    bx = False
    by = False
    bz = False
    if(Axis_of_rotation == 'x' or Axis_of_rotation == 'X'):
        bx = True
        R = Rxrad(Angle_of_rotation)
    if(Axis_of_rotation == 'y' or Axis_of_rotation == 'Y'):
        by = True
        R = Ryrad(Angle_of_rotation)
    if(Axis_of_rotation == 'z' or Axis_of_rotation == 'Z'):
        bz = True
        R = Rzrad(Angle_of_rotation)

    return R

def Displacement(x,y,z):
    P = np.array([[x], [y], [z]])
    return P

def CreateT(R, P):
    H = np.array(
        [
            [R[0,0], R[0,1], R[0,2], P[0,0]],
            [R[1,0], R[1,1], R[1,2], P[1,0]],
            [R[2,0], R[2,1], R[2,2], P[2,0]],
            [0, 0, 0, 1]
        ]
    )
    return H

def getR(T):
    R = np.array(
        [
            [T[0,0], T[0,1], T[0,2]],
            [T[1,0], T[1,1], T[1,2]],
            [T[2,0], T[2,1], T[2,2]]
        ]
    )
    return R

def getP(T):
    P = np.array(
        [
            [T[0,3]],
            [T[1,3]],
            [T[2,3]]
        ]
    )

    return P

def newton_raphson(f, v, x):
    mt = sp.symbols('0')  # The symbolic equivalent of 0
    it_max = 1000  # Maximum number of iterations
    tol = 1e-14  # Tolerance

    if (type(f) == list):
        nofe = len(f)  # Number of equations
        X = np.zeros((it_max, nofe, 1))  # Empty matrix of initial values
        F = np.full((nofe, 1), mt)  # Empty vertical matrix of equations
        V = np.full((nofe), mt)  # Empty horizontal matrix of variables
        for i in range(0, nofe):
            X[0, i, 0] = x[i]  # Filling those three matrices with the input
            F[i, 0] = f[i]
            V[i] = v[i]
    else:
        nofe = 1
        X = np.zeros((it_max, nofe, 1))  # Empty matrix of initial values
        F = np.full((nofe, 1), mt)  # Empty vertical matrix of equations
        V = np.full((nofe), mt)  # Empty horizontal matrix of variables
        for i in range(0, nofe):
            X[0, i, 0] = x  # Filling those three matrices with the input
            F[i, 0] = f
            V[i] = v

    DF = np.full((nofe,nofe), mt)  # An empty square matrix for the Jacobian
    DFX = np.zeros((it_max,nofe,nofe))  # Jacobians with the initial values
    DFX_inv = np.zeros((it_max,nofe,nofe))  # The inverse of that Jacobian
    FX = np.zeros((it_max,nofe,1))  # The input functions of the initial values
    H = np.zeros((it_max,nofe,1))   # A matrix with the residuals

    stop = False    # A boolean variable to end the entire iteration process

    # This loop obtains the derivatives of F
    for i in range(0,nofe):
        for j in range(0,nofe):
            DF[i,j] = sp.diff(F[i,0], V[j])

    # This is where each iteration begins
    for it in range(0,it_max):

        # We start each iteration if all the previous variables don't converge
        if (stop == False):
            conv = 0  # This is how many variables have converged so far
            subby = '{'  # Input for the substitutions done in DFX and FX

            for i in range(0, nofe):
                subby = subby + 'V[' + str(i) + ']: X[it,' + str(i) + ',0]'
                if (i != nofe-1):
                    subby = subby + ',  '
                if (i == nofe-1):
                    subby = subby + '} '

            # Jacobian matrix with the initial values
            for i in range(0,nofe):
                for j in range(0,nofe):
                    DFX[it,i,j] = DF[i,j].subs(eval(subby)).evalf()

            # The input functions with the initial values
            for i in range(0,nofe):
                FX[it,i,0] = F[i,0].subs(eval(subby)).evalf()

            DFX_inv[it] = np.linalg.inv(DFX[it])  # The inverse
            H[it] = np.dot(DFX_inv[it], -FX[it])  # The residuals

            if(it+1 < it_max):
                for i in range(0,nofe):
                    X[it + 1, i, 0] = X[it, i, 0] + H[it, i, 0]  # Update
            else:
                print("It won't converge")
                print("Try with other initial values")
                stop = True  # Stops the loop

            # If each residual is lower than the tolerance, we consider that
            # another variable has reached convergence
            for i in range(0, nofe):
                if(abs(H[it, i, 0]) <= tol):
                    conv = conv + 1

            # If all of the variables have converged we return the final values
            # and stop the loop
            if(conv == nofe):
                if(nofe == 1):
                    return X[it,0,0]
                else:
                    return (X[it,i,0] for i in range(0, nofe))
                stop = True

def StanfordModel(theta, d):
    Rnula = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Pnula = np.array([[0], [0], [0]])
    PInula = np.array([[0], [0], [0], [1]])
    PI0_3 = np.full((3, 1), mt)

    R0_1 = Rotation(theta[0], 'z')
    P0_1 = Displacement(0, 0, d[0])
    T0_1 = CreateT(R0_1, P0_1)

    R1_2z = Rotation(theta[1], 'z')
    R1_2x = Rotation(90, 'x')
    P1_2 = Displacement(0, 0, d[1])

    T1_2x = CreateT(R1_2x, Pnula)
    T1_2z = CreateT(R1_2z, P1_2)
    T1_2 = np.dot(T1_2x, T1_2z)

    T0_2 = np.dot(T0_1, T1_2)

    R2_3x = Rotation(-90, 'x')
    R2_3z = Rotation(theta[2], 'z')
    P2_3 = Displacement(0, 0, d[2])

    T2_3x = CreateT(R2_3x, Pnula)
    T2_3z = CreateT(R2_3z, P2_3)
    T2_3 = np.dot(T2_3x, T2_3z)

    T0_3 = np.dot(T0_2, T2_3)

    R3_4 = Rotation(theta[3], 'z')
    P3_4 = Displacement(0, 0, d[3])
    T3_4 = CreateT(R3_4, P3_4)

    R4_5z = Rotation(theta[4], 'z')
    R4_5x = Rotation(90, 'x')
    P4_5 = Displacement(0, 0, d[4])

    T4_5x = CreateT(R4_5x, Pnula)
    T4_5z = CreateT(R4_5z, P4_5)
    T4_5 = np.dot(T4_5x, T4_5z)

    T3_5 = np.dot(T3_4, T4_5)

    R5_6x = Rotation(-90, 'x')
    R5_6z = Rotation(theta[5], 'z')
    P5_6 = Displacement(0, 0, d[5])

    T5_6x = CreateT(R5_6x, Pnula)
    T5_6z = CreateT(R5_6z, P5_6)
    T5_6 = np.dot(T5_6x, T5_6z)

    T3_6 = np.dot(T3_5, T5_6)

    T0_6 = np.dot(T0_3, T3_6)

    return T0_6

def StanfordModelsym(theta, d):
    Rnula = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Pnula = np.array([[0], [0], [0]])
    PInula = np.array([[0], [0], [0], [1]])
    PI0_3 = np.full((3, 1), mt)

    R0_1 = Rotationrad(theta[0], 'z')
    P0_1 = Displacement(0, 0, d[0])
    T0_1 = CreateT(R0_1, P0_1)

    R1_2z = Rotationrad(theta[1], 'z')
    R1_2x = Rotationrad(sp.pi/2, 'x')
    P1_2 = Displacement(0, 0, d[1])

    T1_2x = CreateT(R1_2x, Pnula)
    T1_2z = CreateT(R1_2z, P1_2)
    T1_2 = np.dot(T1_2x, T1_2z)

    T0_2 = np.dot(T0_1, T1_2)

    R2_3x = Rotationrad(-sp.pi/2, 'x')
    R2_3z = Rotationrad(theta[2], 'z')
    P2_3 = Displacement(0, 0, d[2])

    T2_3x = CreateT(R2_3x, Pnula)
    T2_3z = CreateT(R2_3z, P2_3)
    T2_3 = np.dot(T2_3x, T2_3z)

    T0_3 = np.dot(T0_2, T2_3)

    R3_4 = Rotationrad(theta[3], 'z')
    P3_4 = Displacement(0, 0, d[3])
    T3_4 = CreateT(R3_4, P3_4)

    R4_5z = Rotationrad(theta[4], 'z')
    R4_5x = Rotationrad(sp.pi/2, 'x')
    P4_5 = Displacement(0, 0, d[4])

    T4_5x = CreateT(R4_5x, Pnula)
    T4_5z = CreateT(R4_5z, P4_5)
    T4_5 = np.dot(T4_5x, T4_5z)

    T3_5 = np.dot(T3_4, T4_5)

    R5_6x = Rotationrad(-sp.pi/2, 'x')
    R5_6z = Rotationrad(theta[5], 'z')
    P5_6 = Displacement(0, 0, d[5])

    T5_6x = CreateT(R5_6x, Pnula)
    T5_6z = CreateT(R5_6z, P5_6)
    T5_6 = np.dot(T5_6x, T5_6z)

    T3_6 = np.dot(T3_5, T5_6)

    T0_6 = np.dot(T0_3, T3_6)

    return T0_6

def show(f):
    for i in range(0, 6):
        f = f.replace('sin(theta' + str(i+1) + ')', 's' + str(i+1))
        f = f.replace('cos(theta' + str(i+1) + ')', 'c' + str(i+1))
    return f


theta1, theta2, theta4, theta5, theta6 = sp.symbols('theta1 theta2 theta4 '
                                                    'theta5 theta6')
d1, d2, d3, d6 = sp.symbols('d1 d2 d3 d6')
theta3 = 0
d4, d5 = 0, 0

Theta = [theta1, theta2, theta3, theta4, theta5, theta6]
Dist = [d1, d2, d3, d4, d5, d6]

R0_H = np.array([
                [-0.6597, 0.4160, -0.6258],
                [0.4356, -0.4669, -0.7696],
                [-0.6124, -0.7803, 0.1268]])

PI0_6 = np.array([[-37.5835], [-293.4638], [462.6826], [1]])

Tb_h = StanfordModelsym(Theta, Dist)

for i in range(0,4):
    for j in range(0,4):
        print('T[' + str(i+1) + ',' + str(j+1) + '] = ' + show(str(Tb_h[i,j])))

theta3 = 0
d1, d2, d6 = 400, 200, 100

Theta = [theta1, theta2, theta3, theta4, theta5, theta6]
Dist = [d1, d2, d3, d4, d5, d6]

T0_6 = StanfordModel(Theta, Dist)

f1 = T0_6[0,3] - PI0_6[0,0]
f2 = T0_6[1,3] - PI0_6[1,0]
f3 = T0_6[2,3] - PI0_6[2,0]
f4 = T0_6[2,2] - R0_H[2,2]
f5 = T0_6[1,2] - R0_H[1,2]
f6 = T0_6[2,0] - R0_H[2,0]

f = [f1, f2, f3, f4, f5, f6]
v = [theta1, theta2, d3, theta4, theta5, theta6]
x = (45, 45, 60, 45, 45, 45)

theta1, theta2, d3, theta4, theta5, theta6 = newton_raphson(f, v, x)

print(theta1)
print(theta2)
print(d3)
print(theta4)
print(theta5)
print(theta6)
