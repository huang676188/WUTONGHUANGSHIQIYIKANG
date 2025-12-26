import json
import numpy as np

# 读取数据并处理
with open("makemeahanzi/graphics.txt", "r", encoding="utf-8") as f:
    all_strokes = []
    for line in f:
        data = json.loads(line)
        if data["character"] == "吴":  # 替换为目标汉字
            medians = data["medians"]
            for stroke in medians:
                # 归一化并保留4位小数
                normalized_stroke = [
                    f"np.array([{round(p[0]/3000+0.1,4)}, {round(p[1]/3000+0.1,4)}])"
                    for p in stroke
                ]
                all_strokes.append(normalized_stroke)

# 写入文件（每行一个笔画的np.array列表）
with open("coordinates.txt", "w", encoding="utf-8") as f:
    for stroke in all_strokes:
        f.write("[" + ", ".join(stroke) + "],\n")