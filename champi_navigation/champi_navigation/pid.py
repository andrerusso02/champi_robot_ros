# implement a simple PID controller
import matplotlib.pyplot as plt
import numpy as np
import time

class PID:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.dt = dt
        self.last_time = time.time()

        self.integral = 0
        self.previous_error = 0

        self.error_curve = []

    def update(self, error):
        # self.dt = time.time() - self.last_time
        # self.last_time = time.time()
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        self.error_curve.append(error)

        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
    
    def reset(self):
        self.integral = 0
        self.previous_error = 0


    def display(self, title):
        t = np.arange(0.0, len(self.error_curve)*self.dt, self.dt)
        fig, ax = plt.subplots()
        ax.plot(t, self.error_curve)

        ax.set(xlabel='time (s)', ylabel='error ()',
            title=title)
        ax.grid()

        # fig.savefig("test.png")
        plt.show()


