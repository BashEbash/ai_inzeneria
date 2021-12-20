from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

photo_name="cat.jpg"
result_file="result.jpg"
model_file_name="resnet50_coco_best_v2.1.0.h5"

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , model_file_name))
detector.loadModel()
custom = detector.CustomObjects(cat=True)
detections = detector.detectObjectsFromImage(custom_objects=custom,
                                             input_image=os.path.join(execution_path , photo_name),
                                             output_image_path=os.path.join(execution_path , result_file),
                                             minimum_percentage_probability=50)

if detections:
    print("Its a cat")
else:
    print("Its not a cat")
