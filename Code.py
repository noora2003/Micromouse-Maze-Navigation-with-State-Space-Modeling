import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ load obstacle config ------------------
def load_config(path="walls.json"):
    cfg       = json.loads(Path(path).read_text())
    walls     = [
        (np.array(w["start"], dtype=float),
         np.array(w["end"],   dtype=float))
        for w in cfg["walls"]
    ]
    clearance = float(cfg.get("clearance", 0.05))
    return walls, clearance

# ------------------ Physical & Motor Params ------------------
r, d = 0.02, 0.10
m, I = 0.20, 0.0005
km   = 0.10
b, bθ = 0.01, 0.001

# ------------------ Simulation + Noise Params ------------------
dt, T_max     = 0.01, 10.0
σ_enc, σ_gyro = 0.05, 0.02
Q = np.diag([0.001]*5)
R = np.diag([σ_enc**2, σ_enc**2, σ_gyro**2])

# ------------------ Dynamics & Sensors (for EKF) ------------------
def state_dynamics(x, u):
    _, _, θ, v, ω = x
    Vl, Vr        = u
    return np.array([
        v*np.cos(θ),
        v*np.sin(θ),
        ω,
        (km/(m*r))*(Vl+Vr) - (b/m)*v,
        (d*km/(2*I*r))*(Vr-Vl) - (bθ/I)*ω
    ])

def discretize(x, u):
    return x + dt * state_dynamics(x, u)

def sense(x, rng, noisy=True):
    v, ω = x[3], x[4]
    ωl   = (v + (d/2)*ω)/r
    ωr   = (v - (d/2)*ω)/r
    z    = np.array([ωl, ωr, ω])
    if noisy:
        z += rng.normal(0, [σ_enc, σ_enc, σ_gyro])
    return z

# ------------------ EKF ------------------
def jac_F(x):
    θ, v = x[2], x[3]
    F = np.eye(5)
    F[0,2] = -dt*v*np.sin(θ)
    F[0,3] =  dt*np.cos(θ)
    F[1,2] =  dt*v*np.cos(θ)
    F[1,3] =  dt*np.sin(θ)
    F[2,4] =  dt
    F[3,3] = 1 - dt*(b/m)
    F[4,4] = 1 - dt*(bθ/I)
    return F

def jac_H():
    return np.array([
        [0,0,0, 1/r,  d/(2*r)],
        [0,0,0, 1/r, -d/(2*r)],
        [0,0,0,   0,       1]
    ])

def EKF(x̂, P, y, u, rng):
    x_pred = discretize(x̂, u)
    F      = jac_F(x̂)
    P_pred = F @ P @ F.T + Q

    H      = jac_H()
    y_pred = sense(x_pred, rng, noisy=False)
    S      = H @ P_pred @ H.T + R
    K      = P_pred @ H.T @ np.linalg.inv(S)
    x_upd  = x_pred + K @ (y - y_pred)
    P_upd  = (np.eye(5) - K @ H) @ P_pred
    return x_upd, P_upd

# ------------------ Simple PID Controller ------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp*error + self.ki*self.integral + self.kd*derivative

# ------------------ Reactive Avoidance Controller (multi‐wall) w/ PID ------------------
class ReactiveController:
    """
    Reactive wall‐avoidance + PID controller.
    Moves in a Manhattan pattern toward a goal,
    but spins/backs up when it gets too close to any wall.
    """
    def __init__(
        self,
        walls,            # list of (p1,p2) wall segments as np.array pairs
        clearance,        # how close before we start avoiding
        dt=0.01,          # simulation timestep
        # PID gains: (kp, ki, kd)
        ang_gains=(8, 0.0, -0.1),
        lin_gains=(4, 0.0, 0.6),
        v_max=1.0,        # max forward speed (m/s)
        speed_boost=1.5,  # scales wheel voltages
    ):
        self.walls        = walls
        self.clearance    = clearance
        self.dt           = dt

        # controller state
        self.state        = "go_to_goal"
        self.current_wall = None
        self.turn_target  = None

        # thresholds
        self.angle_eps    = 0.005
        self.align_thresh = 0.02
        self.ω_spin       = np.pi / 2

        # limits & scaling
        self.v_max        = v_max
        self.speed_boost  = speed_boost

        # PID for angle and distance
        self.pid_ang = PID(*ang_gains)
        self.pid_lin = PID(*lin_gains)

    def _find_wall_zone(self, x, y):
        # return the wall segment we're too close to (or None)
        for p1, p2 in self.walls:
            wall_vec  = p2 - p1
            robot_vec = np.array([x, y]) - p1
            t = np.clip(np.dot(robot_vec, wall_vec) /
                        np.dot(wall_vec, wall_vec), 0, 1)
            closest = p1 + t * wall_vec
            if np.linalg.norm([x-closest[0], y-closest[1]]) <= self.clearance:
                return (p1, p2)
        return None

    def to_wheels(self, v_des, ω_des):
        # convert desired linear/angular vel to left/right voltages
        Vl = (v_des - (d/2)*ω_des) * (m*r/km) * self.speed_boost
        Vr = (v_des + (d/2)*ω_des) * (m*r/km) * self.speed_boost
        # clip to physical motor limits [-5V, +5V]
        return np.clip(Vl, -5, 5), np.clip(Vr, -5, 5)

    def control(self, x̂, goal):
        """
        x̂: estimated state [x, y, θ, v, ω]
        goal: [gx, gy]
        returns: Vl, Vr, v_des, ω_des
        """
        x, y, θ = x̂[:3]
        gx, gy  = goal

        def angle_err(target):
            # wrap‐around error in [-π, +π]
            return (target - θ + np.pi) % (2*np.pi) - np.pi

        # 1) WALL DETECTION → spin away
        if self.state == "go_to_goal":
            if wall := self._find_wall_zone(x, y):
                self.current_wall = wall
                self.state        = "turn_away"
                p1, p2            = wall
                # compute outward normal
                normal = np.array([p1[1]-p2[1], p2[0]-p1[0]])
                # pick spin direction so we point away
                self.turn_target = (
                    θ - np.pi/2
                    if np.dot(normal, [np.cos(θ), np.sin(θ)]) > 0
                    else θ + np.pi/2
                )
                self.pid_ang.reset()
                self.pid_lin.reset()
                return 0, 0, 0.0, 0.0

            # 2) MANHATTAN MOVE toward goal
            dx, dy = gx-x, gy-y
            if abs(dy) > self.align_thresh:
                target_angle, dist = (np.pi/2 if dy>0 else -np.pi/2), abs(dy)
            elif abs(dx) > self.align_thresh:
                target_angle, dist = (0.0 if dx>0 else np.pi), abs(dx)
            else:
                # already at goal
                return 0, 0, 0.0, 0.0

            err_ang = angle_err(target_angle)
            ω_des   = self.pid_ang.compute(err_ang, self.dt)

            if abs(err_ang) > self.angle_eps:
                v_des = 0.0
            else:
                v_unclipped = self.pid_lin.compute(dist, self.dt)
                v_des       = np.clip(v_unclipped, -self.v_max, self.v_max)

            Vl, Vr = self.to_wheels(v_des, ω_des)
            return Vl, Vr, v_des, ω_des

        # 3) TURN AWAY from wall until aligned
        if self.state == "turn_away":
            err = angle_err(self.turn_target)
            v_des, ω_des = 0.0, -self.ω_spin
            if abs(err) < self.angle_eps:
                self.state = "clear_wall"
                v_des, ω_des = 0.8, 0.0
            Vl, Vr = self.to_wheels(v_des, ω_des)
            return Vl, Vr, v_des, ω_des

        # 4) CLEAR past the wall
        if self.state == "clear_wall":
            p1, p2 = self.current_wall
            progress = np.dot(
                [x-p1[0], y-p1[1]],
                [np.cos(self.turn_target), np.sin(self.turn_target)]
            )
            if progress > np.linalg.norm(p2-p1) + 0.05:
                self.state       = "turn_to_goal"
                self.turn_target = np.arctan2(gy-y, gx-x)
            v_des, ω_des = 0.8, 0.0
            Vl, Vr = self.to_wheels(v_des, ω_des)
            return Vl, Vr, v_des, ω_des

        # 5) TURN BACK TOWARD goal
        if self.state == "turn_to_goal":
            err = angle_err(self.turn_target)
            v_des, ω_des = 0.0, 2.5 * err
            if abs(err) < self.angle_eps:
                self.state = "go_to_goal"
                self.pid_ang.reset()
                self.pid_lin.reset()
            Vl, Vr = self.to_wheels(v_des, ω_des)
            return Vl, Vr, v_des, ω_des

        # fallback
        return 0, 0, 0.0, 0.0

# ------------------ Simulation ------------------
def simulate(walls, clearance):
    x_true = np.zeros(5)
    x_est  = x_true.copy()
    P      = np.eye(5) * 0.1
    rng    = np.random.default_rng(42)
    ctrl   = ReactiveController(walls, clearance)
    goal   = np.array([0.5, 0.5])

    traj_t, traj_e = [x_true[:2].copy()], [x_est[:2].copy()]
    t = 0.0

    while t < T_max:
        Vl, Vr, v_des, ω_des = ctrl.control(x_est, goal)

        # true motion update
        θ = x_true[2]
        x_true[0] += v_des*np.cos(θ)*dt
        x_true[1] += v_des*np.sin(θ)*dt
        x_true[2] += ω_des*dt
        x_true[3], x_true[4] = v_des, ω_des

        # EKF update
        z       = sense(x_true, rng, noisy=True)
        x_est,P = EKF(x_est, P, z, (Vl, Vr), rng)

        traj_t.append(x_true[:2].copy())
        traj_e.append(x_est[:2].copy())

        if np.hypot(*(x_true[:2] - goal)) < 0.05:
            print(f"🏁 Goal reached in {t:.2f}s")
            break

        t += dt

    return np.array(traj_t), np.array(traj_e), t

# ------------------ Plotting ------------------
def plot_traj(tr_t, tr_e, walls, sim_t):
    plt.figure(figsize=(8,6))
    plt.plot(tr_t[:,0], tr_t[:,1], 'b-', lw=3, label='True Path')
    plt.plot(tr_e[:,0], tr_e[:,1], 'r--', lw=2, label='Est. Path')
    plt.plot(0.5, 0.5, 'g*', ms=15, label='Goal')
    for p1, p2 in walls:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=3)
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f"Reactive Wall Avoidance + PID ({sim_t:.2f}s)")
    plt.legend()
    plt.show()

# ------------------ Main ------------------
if __name__ == "__main__":
    walls, clr    = load_config("walls.json")
    t_tr, e_tr, st = simulate(walls, clr)
    plot_traj(t_tr, e_tr, walls, st)
