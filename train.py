from ultralytics import YOLO

data = './data/cell_yolo/data.yaml'
checkpoint = "./exp/train7/weights/best.pt"

model = YOLO(checkpoint)  # load a pretrained model (recommended for training)

results = model.train(data=data, batch=2, epochs=50, imgsz=640)
#results = model.train(data=data, batch=2, epochs=150, imgsz=640)


# Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category