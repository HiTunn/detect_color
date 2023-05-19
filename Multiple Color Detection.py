# Python code for Multiple Color Detection
'''

'''

import numpy as np
import cv2


# Capturing video through webcam
webcam = cv2.VideoCapture(0)
# path = 'D:/Tunf/NOO/HK1_2022_2023_(Ki1_nam4)/XLA/color_detect/R.jfif'
# path = "D:/Download/cuong2.png"
treshold = 0.85
boxes =[]
def NMS(boxes):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x1 the top-left corner
    y1 = boxes[:, 1]  # y1 the top-left corner
    x2 = boxes[:, 2]  # x2 the bottom-right corner
    y2 = boxes[:, 3]  # y2 the bottom-right corner
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > treshold:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

# Start a while loop
while(1):
	
	# Reading the video from the
	# webcam in image frames
	_, imageFrame = webcam.read()
	# imageFrame = cv2.imread(path)
	# imageFrame = cv2.imread(path)
	# Convert the imageFrame in
	# BGR(RGB color space) to
	# HSV(hue-saturation-value)
	# color space
	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

	# Set range for red color and
	# define mask
	# red_lower = np.array([136, 87, 111], np.uint8)
	# red_lower = np.array([0,50,50], np.uint8)
	# red_upper = np.array([10, 255, 255], np.uint8)
	# red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

	lower_red = np.array([0,50,50], np.uint8)
	upper_red = np.array([10,255,255], np.uint8)
	lower_red2 = np.array([170,50,50], np.uint8)
	upper_red2 = np.array([180,255,255], np.uint8)
	red_mask = cv2.inRange(hsvFrame, lower_red, upper_red) + cv2.inRange(hsvFrame, lower_red2, upper_red2)

	# Set range for green color and
	# define mask
	# green_lower = np.array([25, 52, 72], np.uint8)
	# green_upper = np.array([102, 255, 255], np.uint8)
	
	# green_lower = np.array([40, 50, 70], np.uint8)
	# green_upper = np.array([70, 255, 255], np.uint8)
	# green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

	# Set range for blue color and
	# define mask
	# blue_lower = np.array([94, 80, 2], np.uint8)
	# blue_upper = np.array([120, 255, 255], np.uint8)

	blue_lower = np.array([100,50,50], np.uint8)
	blue_upper = np.array([130,255,255], np.uint8)
	blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

	# Morphological Transform, Dilation
	# for each color and bitwise_and operator
	# between imageFrame and mask determines
	# to detect only that particular color
	kernal = np.ones((5, 5), "uint8")

	# For red color
	red_mask = cv2.dilate(red_mask, kernal)
	res_red = cv2.bitwise_and(imageFrame, imageFrame,
							mask = red_mask)

	# For green color
	# green_mask = cv2.dilate(green_mask, kernal)
	# res_green = cv2.bitwise_and(imageFrame, imageFrame,
	# 							mask = green_mask)

	# For blue color
	blue_mask = cv2.dilate(blue_mask, kernal)
	res_blue = cv2.bitwise_and(imageFrame, imageFrame,
							mask = blue_mask)

	# Creating contour to track red color
	contours, hierarchy = cv2.findContours(red_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 100):
			x, y, w, h = cv2.boundingRect(contour)
			boxes.append([x,y,x+w,y+h])
			
			
			# imageFrame = cv2.rectangle(imageFrame, (x, y),
			# 						(x + w, y + h),
			# 						(0, 0, 255), 2)
			
		
			# cv2.putText(imageFrame, "Red Colour", (x, y),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 1.0,
			# 			(0, 0, 255))
	
	if(len(boxes)>1):
		a = np.array(boxes)
			
		imageFrame = cv2.rectangle(imageFrame,a[0,:2],a[0,2:],
										(0, 0, 255), 2)
		cv2.putText(imageFrame, "Red Colour", a[0,:2],
							cv2.FONT_HERSHEY_SIMPLEX, 1.0,
							(0, 0, 255))
	elif len(boxes) == 1: 
		imageFrame = cv2.rectangle(imageFrame, (boxes[0][0], boxes[0][1]),
								(boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3]),
								(0, 0, 255), 2)
		
	
		cv2.putText(imageFrame, "Red Colour", (boxes[0][0], boxes[0][1]),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0,
					(0, 0, 255))

	else: pass
	# boxes =[]
	# # Creating contour to track green color
	# # contours, hierarchy = cv2.findContours(green_mask,
	# # 									cv2.RETR_TREE,
	# # 									cv2.CHAIN_APPROX_SIMPLE)
	

	# for pic, contour in enumerate(contours):
	# 	area = cv2.contourArea(contour)
	# 	if(area > 5000):
	# 		x, y, w, h = cv2.boundingRect(contour)
	# 		boxes.append([x,y,x+w,y+h])

	# 		# imageFrame = cv2.rectangle(imageFrame, (x, y),
	# 		# 						(x + w, y + h),
	# 		# 						(0, 255, 0), 2)

	# 		# cv2.putText(imageFrame, "Green Colour", (x, y),
	# 		# 			cv2.FONT_HERSHEY_SIMPLEX,
	# 		# 			1.0, (0, 255, 0))
	# if len(boxes)>1:
	# 	a = NMS(np.array(boxes))
	# 	# print(a[0,2:])	
	# 	imageFrame = cv2.rectangle(imageFrame,a[0,:2],a[0,2:],
	# 									(0, 255, 0), 2)
	# 	cv2.putText(imageFrame, "Green Colour", a[0,:2],
	# 						cv2.FONT_HERSHEY_SIMPLEX, 1.0,
	# 						(255, 0, 0))
	# elif len(boxes) == 1:  
	# 	imageFrame = cv2.rectangle(imageFrame, (boxes[0][0], boxes[0][1]),
	# 							(boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3]),
	# 							(0, 255, 0), 2)
		
	
	# 	cv2.putText(imageFrame, "Green Colour", (boxes[0][0], boxes[0][1]),
	# 				cv2.FONT_HERSHEY_SIMPLEX, 1.0,
	# 				(0, 255, 0))
	# else: pass

	boxes =[]

	# Creating contour to track blue color
	contours, hierarchy = cv2.findContours(blue_mask,
										cv2.RETR_TREE,
										cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 100 and area <1000):
			x, y, w, h = cv2.boundingRect(contour)
			boxes.append([x,y,x+w,y+h])
			# imageFrame = cv2.rectangle(imageFrame, (x, y),
			# 						(x + w, y + h),
			# 						(255, 0, 0), 2)
			
			# cv2.putText(imageFrame, "Blue Colour", (x, y),
			# 			cv2.FONT_HERSHEY_SIMPLEX,
			# 			1.0, (255, 0, 0))
	if(len(boxes)>1):
		
		a = np.array(boxes)
		# print(a[0,2:])	
		imageFrame = cv2.rectangle(imageFrame,a[0,:2],a[0,2:],
										(255, 0, 0), 2)
		cv2.putText(imageFrame, "Blue Colour", a[0,:2],
							cv2.FONT_HERSHEY_SIMPLEX, 1.0,
							(255, 0, 0))

	elif len(boxes) == 1: 
		imageFrame = cv2.rectangle(imageFrame, (boxes[0][0], boxes[0][1]),
								(boxes[0][0] + boxes[0][2], boxes[0][1] + boxes[0][3]),
								(255, 0, 0), 2)
		
	
		cv2.putText(imageFrame, "Blue Colour", (boxes[0][0], boxes[0][1]),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0,
					(255, 0, 0))
	else: pass
	boxes =[]
	
	
	
			
	# Program Termination
	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
	cv2.imshow("Multiple Color Detection", red_mask)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	if cv2.waitKey(10) & 0xFF == ord('q'):
		
		# webcam.release()
		cv2.destroyAllWindows()
		break
