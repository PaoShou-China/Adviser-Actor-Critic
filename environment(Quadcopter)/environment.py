from typing import Optional
import os
import numpy as np
from rlschool.quadrotor.quadrotorsim import QuadrotorSim
from .core import Simulator, RobotTask, RobotEnv

NO_DISPLAY = False
try:
    from rlschool.quadrotor.render import RenderWindow
except Exception:
    NO_DISPLAY = True


class QuadrotorSimulator(Simulator):
    def __init__(self, dt: float = 0.01):
        # 调用父类构造函数，初始化仿真时间步长
        super().__init__(dt=dt)

        # 获取配置文件路径
        simulator_conf = os.path.join(os.path.dirname(__file__), 'config.json')

        # 初始化四旋翼模拟器
        self.simulator = QuadrotorSim()

        # 加载配置并重置模拟器
        self.cfg_dict = self.simulator.get_config(simulator_conf)
        self.reset()

        # 初始化控制器
        self.controller = Control()

    def reset(self, initial_state: np.ndarray = None) -> np.ndarray:
        # 根据初始状态重置或默认重置模拟器
        if initial_state is not None:
            self.simulator._restore_state(initial_state)
        else:
            self.simulator.reset()

        # 获取传感器数据和状态信息
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()

        # 构造观察值数组
        observation = np.array([
            state_dict['b_v_x'], state_dict['b_v_y'], state_dict['b_v_z'],
            sensor_dict['acc_x'], sensor_dict['acc_y'], sensor_dict['acc_z'],
            sensor_dict['gyro_x'], sensor_dict['gyro_y'], sensor_dict['gyro_z'],
            sensor_dict['pitch'], sensor_dict['roll'], sensor_dict['yaw'],
        ])
        return observation

    def get_action_dim(self) -> int:
        # 返回动作维度
        return 4

    def step(self, action: np.ndarray) -> np.ndarray:
        # 解析行动指令
        z_vel_sp, pitch_sp, roll_sp, yaw_sp = action

        # 设置控制器的目标姿态角和垂直速度
        self.controller.eul_sp = np.array([roll_sp, pitch_sp, yaw_sp])
        self.controller.z_vel_sp = z_vel_sp

        # 获取当前传感器和状态信息
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()
        current_state = {
            'sensor_dict': sensor_dict,
            'state_dict': state_dict
        }

        # 执行控制循环，得到电机PWM信号
        motor_pwm_signals = self.controller.control_loop(current_state)

        # 更新仿真器状态
        self.simulator.step(motor_pwm_signals, self.dt)

        # 再次获取更新后的传感器和状态信息，构造新的观察值数组
        sensor_dict = self.simulator.get_sensor()
        state_dict = self.simulator.get_state()
        observation = np.array([
            state_dict['b_v_x'], state_dict['b_v_y'], state_dict['b_v_z'],
            sensor_dict['acc_x'], sensor_dict['acc_y'], sensor_dict['acc_z'],
            sensor_dict['gyro_x'], sensor_dict['gyro_y'], sensor_dict['gyro_z'],
            sensor_dict['pitch'], sensor_dict['roll'], sensor_dict['yaw'],
        ])

        return observation


class Control:
    def __init__(self):
        # 初始化设定值
        self.eul_sp = np.zeros(3)
        self.z_vel_sp = 0.0

        # 控制器增益
        self.att_P_gain = np.array([10.0, 10.0, 1.5])  # 姿态P增益
        self.rate_P_gain = np.array([3.0, 3.0, 1.0])  # 角速率P增益
        self.z_vel_P_gain, self.z_vel_forward = 3.0, 5.0  # 垂直速度P增益和前馈项

        # 限制参数
        self.rateMax = np.array([300.0, 300.0, 200.0]) / 57.3  # 最大角速率

    def control_loop(self, state):
        """
        完整的控制循环
        :param state: 当前飞行器状态
        :return: 电机PWM信号
        """
        # 提取传感器数据
        sensor_dict = state['sensor_dict']
        euler = np.array([sensor_dict['roll'], sensor_dict['pitch'], sensor_dict['yaw']])
        rates = np.array([sensor_dict['gyro_x'], sensor_dict['gyro_y'], sensor_dict['gyro_z']])

        # 计算姿态误差
        eul_error = self.eul_sp - euler
        eul_error[2] = (eul_error[2] + np.pi) % (2 * np.pi) - np.pi  # 处理偏航角误差

        # 使用P控制器计算角速率设定值
        pqr_sp = self.att_P_gain * eul_error
        pqr_sp = np.clip(pqr_sp, -self.rateMax, self.rateMax)  # 限制角速率设定值

        # 计算角速率误差
        pqr_error = pqr_sp - rates

        # 使用P控制器计算电机控制信号
        motor_rates = self.rate_P_gain * pqr_error
        z_vel_error = self.z_vel_sp - state['state_dict']['b_v_z']
        thrust = self.z_vel_P_gain * z_vel_error + self.z_vel_forward  # 总推力

        # 混控器分配推力到各个电机
        l = 0.2  # 比例因子
        p, q, r = motor_rates
        T1 = np.clip(thrust + l * (p + q - r), 0.1, 15.0)
        T2 = np.clip(thrust + l * (p - q + r), 0.1, 15.0)
        T3 = np.clip(thrust - l * (p - q - r), 0.1, 15.0)
        T4 = np.clip(thrust - l * (p + q + r), 0.1, 15.0)

        # 返回四个电机的PWM信号
        return np.array([T1, T2, T3, T4])


class QuadrotorTask(RobotTask):

    def get_observation(self) -> np.ndarray:
        sensor_dict = self.sim.simulator.get_sensor()
        state_dict = self.sim.simulator.get_state()
        observation = np.array([
            state_dict['b_v_x'], state_dict['b_v_y'], state_dict['b_v_z'],
            sensor_dict['acc_x'], sensor_dict['acc_y'], sensor_dict['acc_z'],
            sensor_dict['gyro_x'], sensor_dict['gyro_y'], sensor_dict['gyro_z'],
            sensor_dict['pitch'], sensor_dict['roll'], sensor_dict['yaw'],
        ], dtype=np.float32)
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        return np.array([self.sim.simulator.global_velocity[0],
                         self.sim.simulator.global_velocity[1],
                         self.sim.simulator.global_velocity[2]], dtype=np.float32)

    def get_desired_goal(self) -> np.ndarray:
        return self.target_velocity.copy()

    def reset(self, options: dict = None):
        if options is None:
            self.sim.reset()
            self.target_velocity = np.random.uniform(-1.0, 1.0, 3)
        else:
            self.sim.reset(options['init_state'])
            self.target_velocity = options['goal']

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return np.linalg.norm(achieved_goal - desired_goal) < 0.05

    def is_failure(self, observation: np.ndarray, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return False

    def compute_reward(self, action: np.ndarray, observation: np.ndarray, achieved_goal: np.ndarray,
                       desired_goal: np.ndarray, terminated: bool) -> float:
        # 速度差异惩罚
        velocity_diff = np.linalg.norm(achieved_goal - desired_goal)
        task_reward = - velocity_diff
        # 总奖励
        reward = task_reward * self.sim.dt
        if velocity_diff < 0.05: reward += 50

        return reward


class QuadrotorEnv(RobotEnv):

    def render(self):
        if self.render_mode is not None and self.screen is None:
            if NO_DISPLAY:
                raise RuntimeError('[Error] Cannot connect to display screen.')
            self.viewer = RenderWindow(task=self.task)

        state = self._get_state_for_viewer()
        if self.render_mode == 'human':
            self.viewer.view(state, self.task.sim.dt, expected_velocity=self.target_velocity)

    def close(self):
        if self.screen is not None:
            self.screen.close()
            self.screen = None
        super().close()


def make_env(render_mode: str = None) -> QuadrotorEnv:
    simulator = QuadrotorSimulator(dt=0.05)
    task = QuadrotorTask(simulator)
    action_low = np.array([-2.0, -0.2, -0.2, -0.2])
    action_high = np.array([2.0, 0.2, 0.2, 0.2])
    env = QuadrotorEnv(task, action_low, action_high, max_episode_steps=1000, render_mode=render_mode)
    return env


if __name__ == '__main__':
    # 创建环境实例
    env = make_env()

    # 重置环境并获取初始观察
    observation, info = env.reset()
    print(f"Initial observation: {observation}")

    # 执行几个步骤来测试环境
    for _ in range(100):  # 运行 100 步
        # 随机选择一个动作
        action = env.action_space.sample()  # 使用环境定义的动作空间随机采样
        # 在环境中执行动作
        next_observation, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # 如果达到终止条件或被截断，则重置环境
        if terminated or truncated:
            print("Episode finished. Resetting the environment.")
            observation, info = env.reset()
            break

        # 更新当前观察
        observation = next_observation

    # 关闭环境
    env.close()
