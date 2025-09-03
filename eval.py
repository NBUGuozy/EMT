from ultralytics import YOLO

data = './data/cell_yolo/data_eval.yaml'
checkpoint = "./runs/detect/train6/weights/best.pt"

model = YOLO(checkpoint)  # load a pretrained model (recommended for training)


# results = model.train(data=data, epochs=100, imgsz=640)


# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category