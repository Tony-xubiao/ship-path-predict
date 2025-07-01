import pandas as pd
import matplotlib.pyplot as plt


# 读取CSV文件
def read_ship_data(file_path):
    """
    读取船舶轨迹CSV文件
    参数：
        file_path: 数据文件路径（字符串）
    返回：
        包含轨迹数据的DataFrame
    """
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    return df


# 可视化航迹函数
def plot_ship_trajectory(df, title='船舶航迹可视化'):
    """
    绘制船舶航迹图
    参数：
        df: 包含轨迹数据的DataFrame
        title: 图表标题（字符串）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
    plt.rcParams['axes.unicode_minus'] = False

    # 创建画布
    plt.figure(figsize=(12, 8), dpi=100)

    # 绘制航迹线
    plt.plot(df['longitude'],
             df['latitude'],
             linewidth=1.0,
             color='#1f77b4')

    # 标记起点和终点
    plt.scatter(df['longitude'].iloc[0], df['latitude'].iloc[0],
                color='green', s=80, label='起点', zorder=3)
    plt.scatter(df['longitude'].iloc[-1], df['latitude'].iloc[-1],
                color='red', s=80, label='终点', zorder=3)

    # 添加图表元素
    plt.title(title, fontsize=14)
    plt.xlabel('经度 (°E)', fontsize=12)
    plt.ylabel('纬度 (°N)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    plt.legend()

    # 优化坐标轴范围
    plt.xlim(df['longitude'].min() - 0.001, df['longitude'].max() + 0.001)
    plt.ylim(df['latitude'].min() - 0.001, df['latitude'].max() + 0.001)

    # 显示图表
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 读取数据文件（请确保文件路径正确）
    data_file = "wave_ship_data.csv"
    try:
        ship_df = read_ship_data(data_file)
        print(f"成功读取数据文件：{data_file}")
        print(ship_df.head())

        # 可视化航迹
        plot_ship_trajectory(ship_df, title='船舶航迹可视化 (10小时航行)')

    except FileNotFoundError:
        print(f"错误：找不到数据文件 {data_file}")
    except Exception as e:
        print(f"发生错误：{str(e)}")
