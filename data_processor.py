import csv
from tensorboard.backend.event_processing import event_accumulator
import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs('log/processed_log', exist_ok=True)


def find_tfevent_file(directory):
    # 遍历指定目录，寻找扩展名为 .tfevents 的文件
    for filename in os.listdir(directory):
        if filename.startswith('events.out.tfevents.'):
            return os.path.join(directory, filename)
    return None


def extract_tensorboard_data(directory, output_csv, tags):
    # 查找目录中的 .tfevents 文件
    event_file = find_tfevent_file(directory)

    if not event_file:
        print("No .tfevents file found in the directory.")
        return

    # 创建一个 EventAccumulator 对象
    ea = event_accumulator.EventAccumulator(event_file)

    # 加载事件文件
    ea.Reload()

    # 收集所有步数和对应的数据点
    data_points = {}
    max_steps = 0
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            for event in events:
                step = event.step
                if step > max_steps:
                    max_steps = step
                if step not in data_points:
                    data_points[step] = {'step': step}
                data_points[step][tag] = event.value
        else:
            print(f"Tag {tag} not found in the event file.")

    # 打开CSV文件准备写入
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入表头
        header = ['time_step'] + tags
        writer.writerow(header)

        # 写入数据
        for step in range(max_steps + 1):
            row = [data_points.get(step, {}).get(tag, '') for tag in tags]
            writer.writerow([step] + row)


def plot_csv_files(folder_path):
    """
    为指定文件夹中的每个CSV文件绘制独立的图表，包括训练/评估分数和误差。
    每个CSV文件的数据会被绘制在两张不同的图上：一张是分数，另一张是误差。

    参数:
    folder_path (str): 包含CSV文件的文件夹路径。
    """
    # 确保保存图片的目录存在
    os.makedirs('log/pictures', exist_ok=True)

    # 遍历文件夹中的所有csv文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # 构建完整的文件路径
            file_path = os.path.join(folder_path, filename)

            # 读取csv文件
            data = pd.read_csv(file_path)

            # 提取数据
            time_steps = data.iloc[:, 0]
            train_scores = data['score/train']
            eval_scores = data['score/eval']
            train_errors = data['error/train']  # 假设CSV中有这个列
            eval_errors = data['error/eval']    # 假设CSV中有这个列

            # 绘制分数图表
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, train_scores, label='Train Score', color='blue')
            plt.plot(time_steps, eval_scores, label='Eval Score', color='green')
            plt.title(f'Train and Evaluation Scores Over Time Steps - {filename[:-4]}')
            plt.xlabel('Time Steps')
            plt.ylabel('Scores')
            plt.legend()
            scores_output_filename = f'log/pictures/{filename[:-4]}_scores.png'
            plt.savefig(scores_output_filename)
            print(f'Saved {scores_output_filename}')
            plt.close()

            # 绘制误差图表
            plt.figure(figsize=(10, 6))
            plt.plot(time_steps, train_errors, label='Train Error', linestyle='--', color='blue')
            plt.plot(time_steps, eval_errors, label='Eval Error', linestyle='--', color='green')
            plt.title(f'Train and Evaluation Errors Over Time Steps - {filename[:-4]}')
            plt.xlabel('Time Steps')
            plt.ylabel('Errors')
            plt.legend()
            errors_output_filename = f'log/pictures/{filename[:-4]}_errors.png'
            plt.savefig(errors_output_filename)
            print(f'Saved {errors_output_filename}')
            plt.close()


if __name__ == '__main__':
    adviser_train_params_list = [
        [1.0, 0.0, 0.0, 0.1, 0.0],  # naive
        [1.3, 0.0, 0.0, 0.1, 0.0],  # PID1
        [1.3, 0.01, 0.01, 0.1, 0.0],  # PID2
        [0.0, 0.0, 0.0, 0.0, 0.3],  # SF1
        [0.0, 0.0, 0.0, 0.0, 1.0],  # SF2
    ]
    adviser_eval_params_list = [
        [1.0, 0.0, 0.0, 0.1, 0.0],  # naive
        [1.3, 0.0, 0.0, 0.1, 0.0],  # PID1
        [1.3, 0.1, 0.1, 0.1, 0.0],  # PID2
        [0.0, 0.0, 0.0, 0.0, 1.0],  # SF1
        [0.0, 0.0, 0.0, 0.0, 2.0],  # SF2
    ]

    for goal_strategy in ['none', 'future']:
        for adviser_train_params in adviser_train_params_list:
            for adviser_eval_params in adviser_eval_params_list:
                for seed in [1, 2, 3, 4, 5]:
                    filename = f'log/{goal_strategy}/{adviser_train_params}_{adviser_eval_params}/{seed}'
                    directory_path = f'log/{goal_strategy}/{adviser_train_params}_{adviser_eval_params}/{seed}'
                    output_csv_path = f'log/processed_log/{goal_strategy}_{adviser_train_params}_{adviser_eval_params}_{seed}.csv'  # 输出的CSV文件名
                    extract_tensorboard_data(directory_path, output_csv_path,
                                             ['score/train', 'score/eval', 'error/train', 'error/eval'])

    plot_csv_files('log/processed_log')
