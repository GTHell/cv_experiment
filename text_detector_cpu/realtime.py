import cv2
import numpy as np
from inference import TextDetector

# Read the video stream
cap = cv2.VideoCapture(0)

obj = TextDetector()
threshold = 0.5

def test(self, img, threshold=0.5):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # extracting model inputs: batch_size = 1, num_channels = 3 (RGB), height = 704, width = 704
    batch_size, num_channels, height, width = self.network.input_info[self.input_layer].tensor_desc.dims
    
    # resizing the input image to desired size
    resized_image = cv2.resize(img, (width, height))
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    
    # inferene on the input imahe
    result = self.execution_net.infer(inputs={self.input_layer: input_image})
    
    # output predictions
    predictions = result["boxes"]
    # removing no predictions  
    predictions_req = predictions[~np.all(predictions == 0, axis=1)]
    
    self.draw_overlay("Real time", img, resized_image, predictions_req)

    return img

while True:
    ret, frame = cap.read()

    if ret == False:
        continue
    
    frame = obj.test_frame(frame, threshold)

    cv2.imshow("Video Frame", frame)

    # Wait for user input - q, then you will stop the loop
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
