import numpy as np
import cv2

# Testing the size of the image that is being fed to the model over the network

img = cv2.imread('IMG_1411.jpg', 0)

img = cv2.resize(img, (704, 704))

# encode image to bytes and serialize it using messagepack
import msgpack

# encode opencv image as png
_, buffer = cv2.imencode('.png', img)

# serialize with messagepack
packed = msgpack.packb(buffer.tobytes())

# Save the serialized bytes to a file
with open('image.msgpack', 'wb') as outfile:
    outfile.write(packed)