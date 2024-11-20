import pickle as pk
import matplotlib.pyplot as plt


def show_results(path1, path2=None):

    with open(path1, 'rb') as f:

        acc = pk.load(f)

    # with open(path2, 'rb') as f:             # cost
    #     cost = pk.load(f)

    return acc


# BSP = show_results('./Baseline_7640/test_accs')
# topk = show_results('./topk/com_grads_7270/test_accs')
# dgc = show_results('./DGC_results/test_accs')
Our2 = show_results('./results/test_accs')


# print('BSP Max Accuracy: ', max(BSP))
# print('Topk Max Accuracy: ', max(topk))
# print('DGC Max Accuracy: ', max(dgc))
print('Our-2 Max Accuracy', max(Our2))

print()




plt.figure(1, dpi=300)
# plt.plot(BSP, color='blue', label='Baseline', lw=1.2)
# plt.plot(topk, color='green', label='Topk', lw=1.2)
plt.plot(Our2, color='red', label='Our-2', lw=1.2)
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle as pk


def draw(path1, path2):
    with open(path1, 'rb') as f:

        test_accs = pk.load(f)

    with open(path2, 'rb') as f:

        cost = pk.load(f)

    return test_accs, cost


# BSP = show_results('./Baseline_7640/test_accs')
# topk = show_results('./topk/com_grads_7270/test_accs')
# dgc = show_results('./DGC_results/test_accs')
Our2 = show_results('./results/test_accs')


# 绘图
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)

# ax.plot(BSP, color='blue', linestyle='-', linewidth=0.6)
#
# ax.plot(topk, color='green', linestyle='-', linewidth=0.6)
#
# ax.plot(dgc, color='purple', linestyle='-', linewidth=0.6)

ax.plot(Our2, color='red', linestyle='-', linewidth=0.6)

ax.legend(labels=["MP$^{2}$"], ncol=2, prop={'size':6})

plt.xlabel('Number of epochs', size=8)
plt.ylabel('Accuracy', size=8)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.grid(axis='y', color='lightgrey')

# 嵌入绘制局部放大图的坐标系
axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.5, 0.5, 1, 1),
                   bbox_transform=ax.transAxes,
                   borderpad=0.5)

# 在子坐标系中绘制原始数据
# axins.plot(BSP, color='blue', linestyle='-', linewidth=1)
#
# axins.plot(topk, color='green', linestyle='-', linewidth=1)
#
# axins.plot(dgc, color='purple', linestyle='-', linewidth=1)

axins.plot(Our2, color='red', linestyle='-', linewidth=0.6)

# 设置放大区间
# LeNet-5_MNIST
zone_left = 250
zone_right = 299

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0.05 # x轴显示范围的扩展比例
y_ratio = 0.02 # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = zone_left - (zone_right - zone_left)*x_ratio
xlim1 = zone_right + (zone_right - zone_left)*x_ratio
# xlim0 = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
# xlim1 = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio
#
# # Y轴的显示范围

# y = np.hstack((BSP[zone_left:zone_right], topk[zone_left:zone_right], dgc[zone_left:zone_right],
#                Our2[zone_left:zone_right]))
y = np.hstack((Our2[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# y = np.hstack((y_1[zone_left:zone_right], y_2[zone_left:zone_right], y_3[zone_left:zone_right]))
# ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
# ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio
#
# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

tick_params = {'labelsize': 6}  # 设置字体大小为12
axins.tick_params(axis='both', **tick_params)
#
# 建立父坐标系与子坐标系的连接线
# loc1 loc2: 坐标系的四个角
# 1 (右上) 2 (左上) 3(左下) 4(右下)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=1)

# 显示
plt.show()










