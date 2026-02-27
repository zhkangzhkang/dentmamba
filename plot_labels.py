import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置 Matplotlib 使用中文字体
# 自动选择一个系统中支持中文的字体，例如 'SimHei' 或 'Microsoft YaHei'
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK']  # 设置常见中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例绘图
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot([1, 2, 3], [1, 4, 9], label='示例数据')
plt.xlabel('x轴')  # 中文标签
plt.ylabel('y轴')  # 中文标签
plt.title('示例标题')  # 中文标题
plt.legend()

plt.tight_layout()
plt.show()
