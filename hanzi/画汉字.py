import json
import matplotlib.pyplot as plt

# 读取graphics.txt文件
with open("makemeahanzi/graphics.txt", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data["character"] == "吴":  # 替换为目标汉字
            medians = data["medians"]  # 获取笔画中线坐标
            break

# 绘制每个笔画的线条
for stroke in medians:
    x = [point[0] for point in stroke]
    y = [point[1] for point in stroke]
    plt.plot(x, y, marker="o", linestyle="-", markersize=2)

# 设置坐标轴范围（MakeMeAHanzi使用1024x1024坐标系）
plt.xlim(0, 1024)
plt.ylim(0, 1024)
plt.show()