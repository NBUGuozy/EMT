import cv2
import json

# 读取标注文件
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

# 可视化标注
def visualize_annotations(image_path, annotations):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 遍历标注信息
    for annotation in annotations:
        label = annotation["label"]
        coordinates = annotation["coordinates"]
        x, y = int(coordinates["x"]), int(coordinates["y"])
        width, height = int(coordinates["width"]), int(coordinates["height"])

        # 计算边界框的右下角坐标
        x2, y2 = int(x + width/2), int(y + height/2)
        x, y   = int(x - width/2), int(y - height/2)

        # 绘制边界框
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)  # 绿色框，线宽为2

        # 绘制标签
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示图像
    # cv2.imshow("Annotated Image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    output_path = 'visualize_annotations.png'
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved to: {output_path}")

# 主函数
if __name__ == "__main__":
    # 标注文件路径
    annotation_file = "/home/zhangxirui/work/freelance/celldetection/data/raw/i/label/141.json"  # 替换为你的标注文件路径

    # 加载标注文件
    annotations_data = load_annotations(annotation_file)

    # 遍历每张图像的标注
    for item in annotations_data:
        image_path = '/home/zhangxirui/work/freelance/celldetection/data/raw/i/141.tif'  # 图像路径
        annotations = item["annotations"]  # 标注信息

        # 可视化标注
        visualize_annotations(image_path, annotations)