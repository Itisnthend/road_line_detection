import cv2
import numpy as np

import matplotlib.pyplot as plt

from collections import deque

import os

import imageio
imageio.plugins.ffmpeg.download()

from moviepy.editor import VideoFileClip

#selesct yellow and white color
def select_rgb_wy(image):
	lower = np.uint8([230,230,225]) #(G,B,R)
	upper = np.uint8([255,255,255])
	white_mask = cv2.inRange(image,lower,upper)

	lower = np.uint8([160,120,90]) #(G,B,R)
	upper = np.uint8([230,150,120])
	yellow_mask = cv2.inRange(image,lower,upper)

	mask = cv2.bitwise_or(yellow_mask,white_mask)
	masked = cv2.bitwise_and(image,image,mask = mask)
	return masked

#convert rgb into hsl
def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

#select white and yellow color in hsl
def select_hsl_wy(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([0, 200, 0]) #(G,B,R) 10 80 75
    upper = np.uint8([255, 255, 255]) #(G,B,R) 45 255 255
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10, 0,100]) # 10 45 100
    upper = np.uint8([ 40, 255,255]) # 22 240 200
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=40, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

    
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]

    bottom_left  = [cols*0, rows*0.6]
    top_left     = [cols*0.1, rows*0.45]
    bottom_right = [cols*1, rows*0.6]
    top_right    = [cols*0.9, rows*0.45]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)

def hough_lines(image):
    """
    `image` should be the output of a Canny transform.
    
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)



def average_slope_intercept(lists):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    for lines in lists:
        for line in lines:
            x1=line[0]
            y1=line[1]
            x2=line[2]
            y2=line[3]
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                  left_lines.append((slope, intercept))
                  left_weights.append((length))
            else:
                  right_lines.append((slope, intercept))
                  right_weights.append((length))
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
    return left_lane, right_lane # (slope, intercept), (slope, intercept)

          
def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))
    
def lane_lines(image, list_of_lines):
    #for lists in list_of_lines:
    #    for lines in lists:
    #            img= cv2.line(image,(lines[0],lines[1]),(lines[2],lines[3]),(0,0,255),5)
    #            print(lines)
    left_lane, right_lane = average_slope_intercept(list_of_lines)
    
    y1 = image.shape[0]    
    y2 = y1*0.1
    
    new_left_line  = make_line_points(y1, y2, left_lane)
    new_right_line = make_line_points(y1, y2, right_lane)

    return new_left_line, new_right_line

def draw_lane_lines(image, lines, color=[255,0,0],thickness = 20):
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)



#from collections import deque

QUEUE_LENGTH=50

class LaneDetector:
     
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)
     
    def process(self, image):
        #cv2.imshow("image",image)
        white_yellow = select_hsl_wy(image)#select_white_yellow(image)
        #cv2.imshow("image",white_yellow)
        gray         = convert_gray_scale(white_yellow)
        smooth_gray  = apply_smoothing(gray)
        edges        = detect_edges(smooth_gray)
        regions      = select_region(edges)
        lines        = hough_lines(regions)
        left_line, right_line = lane_lines(image, lines)
        '''
        def mean_line(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # make sure it's tuples not numpy array for cv2.line to work
            return line
        
        left_line  = mean_line(left_line,  self.left_lines)
        right_line = mean_line(right_line, self.right_lines)
        '''
        return draw_lane_lines(image, (left_line, right_line))


def process_video(video_input, video_output):
    detector = LaneDetector()

    clip = VideoFileClip(os.path.join('F:\pytest', video_input))
    #clip.preview()
    processed = clip.fl_image(detector.process)
    #processed.write_videofile(os.path.join('F:\pytest', video_output), audio=False)
    processed.preview()


process_video('video.mp4', 'white.mp4')



'''
image = cv2.imread("15image.jpg")

(b,g,r)= cv2.split(image)
rgb_img = cv2.merge([b,g,r]) 

rgb_wy = select_rgb_wy(rgb_img)
hsl_wy = select_hsl_wy(rgb_img)

gray_images = convert_gray_scale(hsl_wy)

blurred_images=apply_smoothing(gray_images)
edge_image=detect_edges(blurred_images)

roi_images= select_region(edge_image)

list_of_lines = hough_lines( roi_images)


for lists in list_of_lines:
        for lines in lists:
                img= cv2.line(image,(lines[0],lines[1]),(lines[2],lines[3]),(255,0,0),5)
                print(lines)


left_lane, right_lane = average_slope_intercept(list_of_lines)


y1 = image.shape[0] 
y2 = y1*0.1 

new_left_line  = make_line_points(y1, y2, left_lane)
new_right_line = make_line_points(y1, y2, right_lane)



#img= cv2.line(image,new_left_line[0] ,new_left_line[1] ,(0,0,255),10)
#img= cv2.line(image,new_right_line[0] ,new_right_line[1] ,(0,0,255),10)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow("image",image)
cv2.waitKey(0)


cv2.destroyAllWindows()
'''
