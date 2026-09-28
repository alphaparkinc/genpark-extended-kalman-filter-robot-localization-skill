"""Example demonstrating EKF localization."""
from client import RobotEKF

def main():
    ekf = RobotEKF(initial_state=(0.0, 0.0, 0.0))
    print("Initial State:", ekf.get_state())
    
    # Predict move
    ekf.predict(v=1.5, omega=0.1, dt=0.5)
    print("After Predict:", ekf.get_state())
    
    # Observe landmark at (5, 5)
    ekf.update(landmark_pos=(5.0, 5.0), measurement=(6.1, 0.72))
    print("After Landmark Update:", ekf.get_state())

if __name__ == "__main__":
    main()
