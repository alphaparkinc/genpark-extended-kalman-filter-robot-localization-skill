"""Extended Kalman Filter (EKF) Robot Localization.
100% Python Standard Library.
"""

import math

def matrix_mult(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, "Dimension mismatch in matrix multiplication"
    return [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]

class RobotEKF:
    """EKF for unicycle mobile robot state [x, y, theta] using landmark range & bearing."""
    def __init__(self, initial_state=(0.0, 0.0, 0.0)):
        self.x = list(initial_state)
        self.P = [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.05]
        ]
        self.Q = [
            [0.05, 0.0, 0.0],
            [0.0, 0.05, 0.0],
            [0.0, 0.0, 0.02]
        ]
        self.R = [
            [0.1, 0.0],
            [0.0, 0.05]
        ]

    def predict(self, v, omega, dt):
        """Kinematic motion update step."""
        theta = self.x[2]
        if abs(omega) < 1e-6:
            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
            dtheta = 0.0
        else:
            dx = -(v / omega) * math.sin(theta) + (v / omega) * math.sin(theta + omega * dt)
            dy = (v / omega) * math.cos(theta) - (v / omega) * math.cos(theta + omega * dt)
            dtheta = omega * dt
            
        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = (self.x[2] + dtheta + math.pi) % (2.0 * math.pi) - math.pi
        
        # Jacobian F_x
        F = [
            [1.0, 0.0, -v * math.sin(theta) * dt],
            [0.0, 1.0,  v * math.cos(theta) * dt],
            [0.0, 0.0, 1.0]
        ]
        FP = matrix_mult(F, self.P)
        F_T = [[F[j][i] for j in range(3)] for i in range(3)]
        FPF_T = matrix_mult(FP, F_T)
        self.P = [[FPF_T[i][j] + self.Q[i][j] for j in range(3)] for i in range(3)]

    def update(self, landmark_pos, measurement):
        """Measurement update with range & bearing measurement = [range, bearing]."""
        lx, ly = landmark_pos
        dx = lx - self.x[0]
        dy = ly - self.x[1]
        q = dx**2 + dy**2
        dist = math.sqrt(q)
        if dist < 1e-4:
            return
            
        expected_range = dist
        expected_bearing = (math.atan2(dy, dx) - self.x[2] + math.pi) % (2.0 * math.pi) - math.pi
        
        meas_range, meas_bearing = measurement
        y_range = meas_range - expected_range
        y_bearing = (meas_bearing - expected_bearing + math.pi) % (2.0 * math.pi) - math.pi
        y = [y_range, y_bearing]
        
        H = [
            [-dx / dist, -dy / dist, 0.0],
            [dy / q,     -dx / q,    -1.0]
        ]
        
        HP = matrix_mult(H, self.P)
        H_T = [[H[j][i] for j in range(2)] for i in range(3)]
        HPH_T = matrix_mult(HP, H_T)
        S = [
            [HPH_T[0][0] + self.R[0][0], HPH_T[0][1] + self.R[0][1]],
            [HPH_T[1][0] + self.R[1][0], HPH_T[1][1] + self.R[1][1]]
        ]
        
        det_S = S[0][0] * S[1][1] - S[0][1] * S[1][0]
        if abs(det_S) < 1e-9:
            return
        inv_S = [
            [ S[1][1] / det_S, -S[0][1] / det_S],
            [-S[1][0] / det_S,  S[0][0] / det_S]
        ]
        
        P_HT = matrix_mult(self.P, H_T)
        K = matrix_mult(P_HT, inv_S)
        
        Ky = [K[0][0] * y[0] + K[0][1] * y[1],
              K[1][0] * y[0] + K[1][1] * y[1],
              K[2][0] * y[0] + K[2][1] * y[1]]
        self.x[0] += Ky[0]
        self.x[1] += Ky[1]
        self.x[2] = (self.x[2] + Ky[2] + math.pi) % (2.0 * math.pi) - math.pi
        
        KH = matrix_mult(K, H)
        I_minus_KH = [[(1.0 if i == j else 0.0) - KH[i][j] for j in range(3)] for i in range(3)]
        self.P = matrix_mult(I_minus_KH, self.P)

    def get_state(self):
        return {
            "x": round(self.x[0], 4),
            "y": round(self.x[1], 4),
            "theta_rad": round(self.x[2], 4),
            "theta_deg": round(math.degrees(self.x[2]), 2),
            "covariance_trace": round(self.P[0][0] + self.P[1][1] + self.P[2][2], 5)
        }
