from ultralytics import YOLO

# Load a model
model = YOLO("./exp/train9/weights/best.pt") # pretrained YOLO11n model

datalist = ['data/cell_yolo/images/train/xue_0008 (2).tif']   # 此处存放推理图片的路径，当前是相对路径
           #'data/cell_yolo/images/train/train_0.jpg'


results = model(datalist, imgsz=640)  # return a list of Results objects

#Process results list
for index, result in enumerate(results):
    boxes = result.boxes
    print(boxes)
    result.save(filename=f"./output/result{index}.jpg")

#from ultralytics import YOLO

# Load a model
#model = YOLO("./exp/train7/weights/best.pt")  # pretrained YOLO11n model

#datapath = 'data/cell_yolo/images/train/shun_0011.tif'
#datalist = [datapath.format(i=i) for i in range(16, 20)]

#results = model.predict(datalist[0], save=True, imgsz=320, conf=0.25, save_txt=True, save_conf=True)

#print(results)
