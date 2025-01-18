import numpy as np

def create_adviser(dim, dt, params):
    kp, ki, kd, T = params 
    adviser = PID(dim=dim, dt=dt, kp=kp, ki=ki, kd=kd, T=T)
    return adviser

class PID:
    def __init__(self, dim, kp=1.0, ki=0.0, kd=0.0, dt=0.01, T=0.1):
        """
        Initialize the PID controller.

        :param dim: Dimension of the error vector.
        :param kp: Proportional gain.
        :param ki: Integral gain.
        :param kd: Derivative gain.
        :param dt: Sampling time.
        :param T: Time constant for derivative term.
        """
        self.dim = dim
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt, self.T = dt, T
        self.reset()

    def reset(self):
        """
        Reset the internal state of the PID controller.
        """
        self.prev_err = np.zeros(self.dim)
        self.int_err = np.zeros(self.dim)

    def __call__(self, obs_err):
        """
        Call the PID controller with the given observation error.

        :param obs_err: The current observation concatenated with the error.
        :return: Updated observation concatenated with the control signal.
        """
        obs_err = np.array(obs_err).flatten()
        obs, err = obs_err[:-self.dim], -obs_err[-self.dim:]
        ctrl_sig = self.run(obs_err)
        return np.concatenate((obs, ctrl_sig))

    def run(self, obs_err):
        """
        Compute the control signal based on the current observation error.

        :param obs_err: The current observation concatenated with the error.
        :return: The control signal.
        """
        obs_err = np.array(obs_err).flatten()
        obs, err = obs_err[:-self.dim], -obs_err[-self.dim:]

        # Update integral error
        self.int_err += err * self.dt

        # Compute derivative error
        deriv_err = (err - self.prev_err) / self.T

        # Update previous error for next iteration
        self.prev_err += (err - self.prev_err) * self.dt / self.T

        # Compute control signal
        ctrl_sig = -self.kp * err - self.ki * self.int_err - self.kd * deriv_err

        return ctrl_sig
