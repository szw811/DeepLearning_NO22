# import matplotlib.pyplot as plt
# import numpy as np
# np.random.seed(4)
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# x = ['aaaa','bbbb','cccc','dddd','eeee']
# y1 = np.random.rand(5)
# y2 = np.random.rand(5)
# plot1 = ax1.plot(range(0, len(x)), y1, '-*', color='r', label='train')
# ax2 = ax1.twinx()  # this is the important function
#
# plot2 = ax2.plot(range(0, len(x)), y2, '-o', color='g', label='test')
# lines = plot1 + plot2
#
# for tl in ax1.get_yticklabels():
#     tl.set_color('r')
# for tl in ax1.get_xticklabels():
#     tl.set_rotation(45)
#     tl.set_fontsize(8)
# for tl in ax2.get_yticklabels():
#     tl.set_color('g')
#
# # 设置坐标轴的标签
# # ax1.set_ylabel('imbanlance', fontsize=15)
# # ax2.set_xlabel('attributes', fontsize=15)
# # ax2.set_ylabel('ratio', fontsize=15)
#
# ax1.legend(lines, [l.get_label() for l in lines]) # only need one legend definition
# plt.show()

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为微软雅黑

# 定义房价
price = [10000, 12000, 16000, 14000, 13000, 19000]
# 定义成交量
total = [100, 50, 40, 60, 120, 40]
# 定义年份
year = [2010, 2011, 2012, 2013, 2014, 2015]

fig, ax = plt.subplots(1, 1)
# 共享x轴，生成次坐标轴
ax_sub = ax.twinx()
# 绘图
# l1, = ax.plot(year, price, 'r-', label='price')
# l2, = ax_sub.plot(year, total, 'g-', label='total')
l1, = ax.plot(price, 'r-')
l2, = ax_sub.plot(total, 'g-')
# 放置图例
plt.legend(handles=[l1, l2], labels=['price', 'total'], loc=0)
# 设置主次y轴的title
ax.set_ylabel('房价(元)')
ax_sub.set_ylabel('成交量(套)')
# 设置x轴title
ax.set_xlabel('年份')
# 设置图片title
ax.set_title('主次坐标轴演示图')
plt.show()

