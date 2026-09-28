# Extended Kalman Filter (EKF) Localization Skill

Nonlinear Extended Kalman Filter for autonomous ground robot state estimation, dead reckoning odometry fusion, and range-bearing landmark localization.

```mermaid
flowchart TD
    Odometry["Control / Odometry (v, ω)"] --> Predict["Prediction Step (x_k|k-1, P_k|k-1)"]
    Landmark["Landmark Observation (r, φ)"] --> Update["Measurement Update Step"]
    Predict --> Update
    Update --> Gain["Compute Kalman Gain K"]
    Gain --> Correct["State Correction x_k|k & Covariance P_k|k"]
```

## Features
- **100% Python Standard Library**: Pure matrix algebra and trigonometric projections.
- **Unicycle Kinematics**: First-order Taylor linearization of differential drive models.
- **Multi-Sensor Fusion**: Combines wheel speed and range-bearing beacon updates.
