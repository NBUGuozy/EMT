import cv2
import os

def visualize_yolo_labels(image_path, label_path, class_names=None):
    """
    可视化 YOLO 格式的标签。

    参数:
        image_path (str): 图片文件路径。
        label_path (str): 标签文件路径。
        class_names (list): 类别名称列表，默认为 None。
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return

    # 获取图片尺寸
    img_height, img_width = image.shape[:2]

    # 读取标签文件
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # 解析每一行标签
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            print(f"警告：跳过无效的标签行 {line}")
            continue

        # 解析 YOLO 格式
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        score = float(parts[5])

        # 转换为边界框坐标
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 绘制检测框
        color = (0, 255, 0)  # 绿色框
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        #这段代码使用 OpenCV 库在图像上绘制一个矩形框。让我们逐行解析这段代码：

#**`color = (0, 255, 0)`**:
#这段代码中，绘制的矩形框是绿色的，因为指定的颜色为 `color = (0, 255, 0)`。如果需要不同颜色的标签框，只需更改 `color` 的值即可。例如：
  -#**蓝色**: `(255, 0, 0)`
  -#**红色**: `(0, 0, 255)`
  -#**白色**: `(255, 255, 255)`
  -#**黑色**: `(0, 0, 0)`

        # 获取标签名称
        if class_names is not None and class_id < len(class_names):
            label = class_names[class_id] + "  {:.2%}".format(score)
        else:
            label = str(class_id) + "  {:.2%}".format(score)

        # 在框的左上角显示标签名称
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

        # 计算文本位置（左上角，框的上方）
        text_x = x1
        text_y = y1 - 5  # 将文本放在框的上方，留出 5 像素的间距

        # 如果文本超出图片顶部，调整位置
        if text_y < 0:
            text_y = y1 + text_size[1] + 5  # 将文本放在框的内部顶部

        # 绘制文本背景（增强可读性）
        cv2.rectangle(
            image,
            (text_x, text_y - text_size[1]),
            (text_x + text_size[0], text_y),
            color,
            -1,  # 填充矩形
        )

        # 绘制标签文本
        cv2.putText(image, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)


    # 显示图片
    cv2.imwrite("./vis.jpg", image)


# 示例用法
if __name__ == "__main__":
    # 设置图片和标签路径
    image_path = "/home/SENSETIME/zhangxirui1/桌面/vision/cell_yolo/images/train/0.jpg"  # 替换为你的图片路径
    label_path = "/home/SENSETIME/zhangxirui1/桌面/vision/cell_yolo/0.txt"  # 替换为你的标签路径

    # 类别名称列表（根据你的数据集修改）
    class_names = ["i", "ii", "iii", "iv", "v"]

    # 可视化标签
    visualize_yolo_labels(image_path, label_path, class_names)