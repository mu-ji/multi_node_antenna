"""
离线数据处理脚本 - 数据读取
用于读取保存在 three_rx_experiment 文件夹下的实验数据
"""

import contextlib
import io

import numpy as np
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

import cmath

SPEED_OF_LIGHT  = 299792458

# 图例 / 色条文字大小（pt），按需修改
LEGEND_FONTSIZE_AOA_RESULT = 10
LEGEND_FONTSIZE_SLOTF = 10
LEGEND_FONTSIZE_HEATMAP_CBAR = 15

# 坐标轴标签 xlabel / ylabel 文字大小（pt），按需修改
AXIS_LABEL_FONTSIZE_AOA_RESULT = 15
AXIS_LABEL_FONTSIZE_SLOTF = 15
AXIS_LABEL_FONTSIZE_HEATMAP = 15

# 坐标轴刻度数字大小（pt），按需修改
TICK_LABEL_FONTSIZE_AOA_RESULT = 10
TICK_LABEL_FONTSIZE_SLOTF = 15
TICK_LABEL_FONTSIZE_HEATMAP = 11


def load_data(filename):
    """
    加载保存的实验数据
    
    参数:
        filename: 数据文件路径 
    
    返回:
        data_dict: 包含所有数据的字典
            - angle_list: 角度列表
            - num_triggers: 触发次数
            - triggers: 每次触发的数据列表
                - index: 触发索引
                - angle: 角度
                - rx1_data: 接收端1的数据
                - rx2_data: 接收端2的数据
                - rx3_data: 接收端3的数据
    """
    data = np.load(filename, allow_pickle=True)
    
    # 提取基本信息
    angle_list = data['angle_list']
    num_triggers = int(data['num_triggers'])
    
    # 组织每次触发的数据
    triggers = []
    for idx in range(num_triggers):
        prefix = f'trigger_{idx:03d}'
        trigger_data = {
            'index': idx,
            'angle': float(data[f'{prefix}_angle']),
            'rx1_data': {
                'interval': int(data[f'{prefix}_rx1_interval']),
                'tx1_sqn': int(data[f'{prefix}_rx1_tx1_sqn']),
                'tx2_sqn': int(data[f'{prefix}_rx1_tx2_sqn']),
                'packet_1_I_data': data[f'{prefix}_rx1_pkt1_I'],
                'packet_1_Q_data': data[f'{prefix}_rx1_pkt1_Q'],
                'packet_2_I_data': data[f'{prefix}_rx1_pkt2_I'],
                'packet_2_Q_data': data[f'{prefix}_rx1_pkt2_Q'],
            },
            'rx2_data': {
                'interval': int(data[f'{prefix}_rx2_interval']),
                'tx1_sqn': int(data[f'{prefix}_rx2_tx1_sqn']),
                'tx2_sqn': int(data[f'{prefix}_rx2_tx2_sqn']),
                'packet_1_I_data': data[f'{prefix}_rx2_pkt1_I'],
                'packet_1_Q_data': data[f'{prefix}_rx2_pkt1_Q'],
                'packet_2_I_data': data[f'{prefix}_rx2_pkt2_I'],
                'packet_2_Q_data': data[f'{prefix}_rx2_pkt2_Q'],
            }
        }
        triggers.append(trigger_data)
    
    return {
        'angle_list': angle_list,
        'num_triggers': num_triggers,
        'triggers': triggers
    }


def calculate_angle(I1, Q1, I2, Q2):
    """
    计算第一个IQ向量到第二个IQ向量之间的夹角
    
    参数:
        I1, Q1: 第一个IQ向量的I和Q分量（可以是标量或数组）
        I2, Q2: 第二个IQ向量的I和Q分量（可以是标量或数组）
    
    返回:
        theta: 夹角（弧度），范围 [-π, π]
               如果输入是数组，返回对应位置的夹角数组
    """
    # 转换为numpy数组以确保兼容性
    I1 = np.asarray(I1)
    Q1 = np.asarray(Q1)
    I2 = np.asarray(I2)
    Q2 = np.asarray(Q2)
    
    # 计算两个向量的相位
    phase1 = np.arctan2(Q1, I1) #- 0.5*np.pi  # 第一个向量的相位
    phase2 = np.arctan2(Q2, I2)  # 第二个向量的相位
    
    # 计算相位差（从phase1到phase2）
    theta = phase1 - phase2
    
    # 将角度约束到 [-π, π] 范围
    theta = np.arctan2(np.sin(theta), np.cos(theta))
    
    return theta

def cal_slope(phase_diff):
    x = np.arange(len(phase_diff)).reshape(-1, 1)
    y = phase_diff
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_

def  phase_to_angle(geo):
    # val = geo/(2*np.pi)*12.5/3.75
    val = geo/(2*np.pi)*12.5/4.3
    if val > 1:
        val = 1
    if val < -1:
        val = -1
    return np.arcsin(val)/np.pi*180
    
def unwrap_and_adjust(numpy_array):

    unwrapped = np.unwrap(numpy_array)
    adjusted_array = np.arctan2(np.sin(unwrapped), np.cos(unwrapped))
    return adjusted_array

def angle_to_phase(angle):
    return angle/180*np.pi*4.3/12.5*2*np.pi
def compensate_phase_offset(trigger_data, ref_position, pos):
    # 提取三个接收端的数据
    rx1_data = trigger_data['rx1_data']
    rx2_data = trigger_data['rx2_data']
    
    # 计算每个接收端每个包的相位
    # RX1
    rx1_pkt1_phase = np.arctan2(rx1_data['packet_1_Q_data'], rx1_data['packet_1_I_data']) #+ 468/360*2*np.pi
    rx1_pkt2_phase = np.arctan2(rx1_data['packet_2_Q_data'], rx1_data['packet_2_I_data']) #+ 468/360*2*np.pi
    
    # RX2
    rx2_pkt1_phase = np.arctan2(rx2_data['packet_1_Q_data'], rx2_data['packet_1_I_data'])
    rx2_pkt2_phase = np.arctan2(rx2_data['packet_2_Q_data'], rx2_data['packet_2_I_data'])
    
    interval = (rx1_data['interval'] + rx2_data['interval'])/2
    # 计算每个接收端每个包的相位
    # RX1

    pkt1_phase_diff_12 = calculate_angle(rx1_data['packet_1_I_data'], rx1_data['packet_1_Q_data'], rx2_data['packet_1_I_data'], rx2_data['packet_1_Q_data'])

    pkt2_phase_diff_12 = calculate_angle(rx1_data['packet_2_I_data'], rx1_data['packet_2_Q_data'], rx2_data['packet_2_I_data'], rx2_data['packet_2_Q_data'])

    print('pkt1_phase_diff_12, pkt2_phase_diff_12:', pkt1_phase_diff_12[0], pkt2_phase_diff_12[0])
    pkt1_phase_diff_12_unwrap = np.unwrap(pkt1_phase_diff_12)
    pkt2_phase_diff_12_unwrap = np.unwrap(pkt2_phase_diff_12)

    # pkt1_phase_diff_12 = np.unwrap(pkt1_phase_diff_12)
    # pkt2_phase_diff_12 = np.unwrap(pkt2_phase_diff_12)

    slope_12, intercept_12 = cal_slope(pkt1_phase_diff_12_unwrap)

    print('slope_12:', slope_12)
    print('interval:', interval)
    print('interval:', slope_12 * interval/16)
    anchor_phase = angle_to_phase(ref_position)

    target_phase_diff_12 = np.arctan2(np.sin(pkt2_phase_diff_12 - pkt1_phase_diff_12 - slope_12 * interval/16 - anchor_phase), np.cos(pkt2_phase_diff_12 - pkt1_phase_diff_12 - slope_12 * interval/16 - anchor_phase))
    # target_phase_diff_12 = np.arctan2(np.sin(pkt1_phase_diff_12 - pkt2_phase_diff_12), np.cos(pkt1_phase_diff_12 - pkt2_phase_diff_12))

    # #-----------------------------------
    # local_phase_shift = np.arctan2(np.sin(pkt1_phase_diff_12 + anchor_phase), np.cos(pkt1_phase_diff_12 + anchor_phase))
    # local_phase_shift = np.arctan2(np.sin(local_phase_shift + slope_12 * interval/16), np.cos(local_phase_shift + slope_12 * interval/16))
    # target_geo_phase_shift = np.arctan2(np.sin(pkt2_phase_diff_12 - local_phase_shift), np.cos(pkt2_phase_diff_12 - local_phase_shift))
    # target_phase_diff_12 = target_geo_phase_shift
    # #-----------------------------------
    print(target_phase_diff_12)
    target_phase_diff_12_unwrap = unwrap_and_adjust(target_phase_diff_12)

    phase_12 = np.mean(target_phase_diff_12_unwrap)

    array1 = unwrap_and_adjust(rx1_pkt1_phase - rx1_pkt2_phase)
    array2 = unwrap_and_adjust(rx2_pkt1_phase - rx2_pkt2_phase)
    array12 = unwrap_and_adjust(array1 - array2)

    # if abs(np.mean(array12)) < np.pi: 
    #     if np.mean(array12) < 0:
    #         return phase_to_angle(phase_12 - np.pi)
    #     else:
    #         return phase_to_angle(phase_12)
    # else:
    #     return phase_to_angle(np.arctan2(np.sin(phase_12 - np.pi), np.cos(phase_12 - np.pi)))

    # rx1_amplitude = np.mean(np.sqrt(rx1_data['packet_2_I_data']**2 + rx1_data['packet_2_Q_data']**2))
    # rx2_amplitude = np.mean(np.sqrt(rx2_data['packet_2_I_data']**2 + rx2_data['packet_2_Q_data']**2))


    if pos > 0:
        if phase_12 > 0:
            return phase_to_angle(np.arctan2(np.sin(phase_12 - np.pi), np.cos(phase_12 - np.pi)))
        else:
            return phase_to_angle(np.arctan2(np.sin(phase_12), np.cos(phase_12)))
    elif pos == 0:
        return phase_to_angle(phase_12)
    else:
        if phase_12 < 0:
            return phase_to_angle(np.arctan2(np.sin(phase_12 + np.pi), np.cos(phase_12 + np.pi)))
        else:
            return phase_to_angle(np.arctan2(np.sin(phase_12), np.cos(phase_12)))


    return phase_to_angle(phase_12)


def filter_array(arr):
    arr = np.array(arr)
    value_to_remove = 90
    mask = (arr != value_to_remove) & (arr != -90)
    print(mask)
    new_arr = arr[mask]
    return new_arr


def heatmap_anchor_angle_estimation(position_list, data_file_tx1a, anchor_a):
    """
    热力图用：对给定 anchor_a（补偿参考），在 position_list 每个 pos 上计算
    各次触发、取反后的角度估计，经 filter_array 后取中位数（°）；无有效样本则为 nan。
    """
    out = []
    for pos in position_list:
        filename = (
            'discrete_antenna_experiment/tx1d_30_tx1a_{}_tx2d_40_tx2a_{}.npz'
        ).format(data_file_tx1a, pos)
        data_dict = load_data(filename)
        est_list = [
            -compensate_phase_offset(
                data_dict['triggers'][i], anchor_a, pos
            )
            for i in range(len(data_dict['triggers']))
        ]
        new_arr = filter_array(est_list)
        if new_arr.size > 0:
            out.append(float(np.mean(new_arr)))
        else:
            out.append(float('nan'))
    return out


# 使用示例：
if __name__ == '__main__':

    # 热力图：是否在格子上标注绝对误差数值
    show_heatmap_cell_values = True

    # 各图图例 / 色条文字大小（pt），按需修改
    legend_fontsize_aoa = 11
    legend_fontsize_heatmap_cbar = 10

    position_list = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80]
    all_phase_data = []
    # position_list = [-20, -10, 10, 20, 30]
    # 收集所有数据
    for pos in position_list:
        anchor_angle = 20
        filename = 'discrete_antenna_experiment/tx1d_30_tx1a_{}_tx2d_40_tx2a_{}.npz'.format(anchor_angle, pos)
        # filename = 'discrete_antenna_experiment/angle_{}.npz'.format(pos)
        data_dict = load_data(filename)
        print(f"成功加载 {data_dict['num_triggers']} 次触发数据")
        print(f"角度列表: {data_dict['angle_list']}")
        
        phase12_list = []
        for i in range(len(data_dict['triggers'])):
            trigger = data_dict['triggers'][i]
            phase12 = -compensate_phase_offset(trigger, anchor_angle, pos)
            phase12_list.append(phase12)
        
        all_phase_data.append(phase12_list)


    print(all_phase_data[0])
    plt.figure()

    data_for_violin = [
        filter_array(all_phase_data[i]) for i in range(len(all_phase_data))
    ]
    # 相邻角度间隔 10°，宽度略小于间隔以免重叠
    violin_width = 7.0
    violin_color = '#4C72B0'

    # parts = plt.violinplot(
    #     data_for_violin,
    #     positions=position_list,
    #     widths=violin_width,
    #     showmeans=False,
    #     showmedians=True,
    # )
    # for pc in parts['bodies']:
    #     pc.set_facecolor(violin_color)
    #     pc.set_edgecolor(violin_color)
    #     pc.set_alpha(0.65)
    # for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
    #     if key in parts:
    #         parts[key].set_edgecolor(violin_color)
    #         parts[key].set_linewidth(1.0)

    bp = plt.boxplot(
        data_for_violin,
        positions=position_list,
        widths=violin_width * 0.7,
        patch_artist=True,
        manage_ticks=False,
    )
    for patch in bp['boxes']:
        patch.set_facecolor('white')
        patch.set_edgecolor(violin_color)
        patch.set_linewidth(1.0)
    for whisker in bp['whiskers']:
        whisker.set_color(violin_color)
    for cap in bp['caps']:
        cap.set_color(violin_color)
    for median in bp['medians']:
        median.set_color(violin_color)

    # 横轴为真实角度（°）；Ground truth：y = x
    plt.plot([-80, 80], [-80, 80], 'r--')
    plt.xlim(-90, 90)
    plt.xticks(list(range(-80, 81, 10)))
    plt.ylim(-100, 100)
    plt.yticks(list(range(-100, 101, 10)))
    plt.gca().tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE_AOA_RESULT)
    plt.xlabel('True angle (°)', fontsize=AXIS_LABEL_FONTSIZE_AOA_RESULT)
    plt.ylabel('Estimated angle (°)', fontsize=AXIS_LABEL_FONTSIZE_AOA_RESULT)
    plt.legend(
        handles=[
            Patch(
                facecolor='white',
                edgecolor=violin_color,
                alpha=0.65,
                label='Angle estimation',
            ),
            Line2D(
                [0],
                [0],
                color='red',
                linestyle='--',
                linewidth=1.5,
                label='Ground truth',
            ),
        ],
        fontsize=LEGEND_FONTSIZE_AOA_RESULT,
    )
    plt.grid()
    plt.savefig('AoA result.svg')
    plt.show()

    # ---------- 热力图：每个 anchor 单独算估计，变量 anchor_*_estimation；再合并 matrix 绘图 ----------
    # x：真实角度 pos；y：anchor_angle；热力图颜色：绝对误差 |估计中位数 − 理想值|（°）。
    # 估计已对 compensate 取反；理想值与第一张图红线一致：ideal = pos。
    anchor_angle_list = list(range(-30, 31, 10))
    data_file_tx1a = 0

    with contextlib.redirect_stdout(io.StringIO()):
        anchor_neg30_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, -30
        )
        anchor_neg20_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, -20
        )
        anchor_neg10_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, -10
        )
        anchor_0_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, 0
        )
        anchor_pos10_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, 10
        )
        anchor_pos20_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, 20
        )
        anchor_pos30_estimation = heatmap_anchor_angle_estimation(
            position_list, data_file_tx1a, 30
        )

    # 合并为 ndarray，再算绝对误差矩阵（仅用于 imshow / 标注）
    estimate_matrix = np.asarray(
        [
            anchor_neg30_estimation,
            anchor_neg20_estimation,
            anchor_neg10_estimation,
            anchor_0_estimation,
            anchor_pos10_estimation,
            anchor_pos20_estimation,
            anchor_pos30_estimation,
        ],
        dtype=float,
    )

    estimate_matrix = np.asarray(
        [
            anchor_pos30_estimation,
            anchor_pos20_estimation,
            anchor_pos10_estimation,
            anchor_0_estimation,
            anchor_neg10_estimation,
            anchor_neg20_estimation,
            anchor_neg30_estimation,
        ],
        dtype=float,
    )
    position_arr = np.asarray(position_list, dtype=float)
    error_matrix = np.abs(estimate_matrix - position_arr[np.newaxis, :])

    pos_step = position_list[1] - position_list[0] if len(position_list) > 1 else 10.0
    anchor_step = (
        anchor_angle_list[1] - anchor_angle_list[0]
        if len(anchor_angle_list) > 1
        else 10.0
    )
    extent = [
        position_list[0] - pos_step / 2,
        position_list[-1] + pos_step / 2,
        anchor_angle_list[0] - anchor_step / 2,
        anchor_angle_list[-1] + anchor_step / 2,
    ]

    # 4) 绘制热力图（error_matrix）
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        error_matrix,
        aspect='auto',
        origin='lower',
        extent=extent,
        interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('error (°)', fontsize=LEGEND_FONTSIZE_HEATMAP_CBAR)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE_HEATMAP)
    ax.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE_HEATMAP)
    ax.set_xlabel('Target angle (°)', fontsize=AXIS_LABEL_FONTSIZE_HEATMAP)
    ax.set_ylabel('Sub-anchor angle (°)', fontsize=AXIS_LABEL_FONTSIZE_HEATMAP)

    if show_heatmap_cell_values:
        for ai in range(error_matrix.shape[0]):
            for pj in range(error_matrix.shape[1]):
                v = error_matrix[ai, pj]
                if np.isnan(v):
                    label = ''
                else:
                    label = f'{v:.2f}'
                t = ax.text(
                    position_list[pj],
                    anchor_angle_list[ai],
                    label,
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white',
                )
                t.set_path_effects(
                    [pe.withStroke(linewidth=2, foreground='black')]
                )

    fig.tight_layout()
    plt.savefig('error_heatmap.svg')
    plt.show()