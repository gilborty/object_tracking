from __future__ import division

import numpy as np
import argparse
import imutils
import cv2
import collections

import debug_status as ds
import roi_selector as rs

threshold_values = collections.namedtuple('threshold_alues', ['low_h', 'low_s', 'low_v', 'high_h', 'high_s', 'high_v'])

debug_mode = True

def compute_stdDev(values):
    """
        Computes the standard deviation of a list
    """
    return np.std(values)

def compute_mask(img):
    """
        Computes an adaptive threshold mask for use in camshift
    """
    #Split the channels
    hsv_frame = img.copy()
    hsv = cv2.split(hsv_frame)
    h,s,v = cv2.split(hsv_frame)

    h_pixels = []
    s_pixels = []
    v_pixels = []

    for r in xrange(0, len(hsv[0])):
        for c in xrange(0, len(hsv[0][r])):
            #If it is a zero pixel, don't add it to the mean
            h_value = hsv[0][r][c]
            s_value = hsv[1][r][c]
            v_value = hsv[2][r][c]
            if h_value == 0 and s_value == 0 and v_value == 0:
                #ds.print_status(ds.FATAL_ERROR, "Zero Pixel")
                continue
            else:
                h_pixels.append(h_value)
                s_pixels.append(s_value)
                v_pixels.append(v_value)
                
    #Compute std_dev
    h_mean = sum(h_pixels)/len(h_pixels)
    s_mean = sum(s_pixels)/len(s_pixels)
    v_mean = sum(v_pixels)/len(v_pixels)

    h_std = compute_stdDev(h_pixels)
    s_std = compute_stdDev(s_pixels)
    v_std = compute_stdDev(v_pixels)

    while True:
        std_ratio_h = 2.0
        std_ratio_s = 2.0
        std_ratio_v = 1.0   

        low_h = (h_mean - h_std*std_ratio_h)
        high_h = (h_mean + h_std*std_ratio_h)
        if low_h <= 0.0:
            low_h = 0.0
        if high_h >= 180:
            high_h = 180.    
        
        low_s = (s_mean - s_std*std_ratio_s)
        high_s = (s_mean + s_std*std_ratio_s)
        if low_s <= 0.0:
            low_s = 0.0
        if high_s >= 255:
            high_s = 255.
        
        
        low_v = (v_mean - v_std*std_ratio_v)
        high_v = (v_mean + v_std*std_ratio_v)
        if low_v <= 0.0:
            low_v = 0.0
        if high_v >= 255:
            high_v = 255.
        

        # mask = cv2.inRange(hsv_frame, 
        #                 np.array((low_h, low_s,low_v)), 
        #                 np.array((high_h,high_s,high_v)))
        # cv2.imshow("Std mask", mask)
        # key = cv2.waitKey(10)
        # if key == 27:
        #     break
        break

        
    return threshold_values(low_h, low_s, low_v, high_h, high_s, high_v)



def resize_frame(img, ratio=2):
    """
        Resizes a frame based on a given ratio. i.e. ratio = 2 would be a 50% reduction in size
    """
    frame_height, frame_width = img.shape[:2]
    cv2.resize(img, (int(frame_width/ratio), int(frame_height/ratio)), interpolation=cv2.INTER_CUBIC)

def find_largest_contour(img):
    """
        Finds the largest contour in the image
    """
    gray_img = img.copy()
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray_img,1,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #If we found at least one contours
    if len(contours) >= 1:
        #Find the largest contour
        largest_area = 0
        largest_contour_index = 0
        for i in xrange(0, len(contours)):
            area = cv2.contourArea(contours[i])
            if area > largest_area:
                largest_area = area
                largest_contour_index = i
    
        cnt = contours[largest_contour_index]
        x,y,w,h = cv2.boundingRect(cnt)
        return img[y:y+h,x:x+w]
    else:
        ds.print_status(ds.FATAL_ERROR, "No contours found")
        return img
    

def get_percentage_of_zero_pixels(img):
    """
        returns decimal percentage of zero pixels in an image
    """
    gray_img = img.copy()
    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    zero_pixels = gray_img.size - cv2.countNonZero(gray_img)
    percentage = zero_pixels/gray_img.size 
    
    if debug_mode:
        ds.print_status(ds.WARNING, "Total pixels: %s" % gray_img.size)
        ds.print_status(ds.WARNING, "Zero pixels: %s" % zero_pixels)
        ds.print_status(ds.WARNING, "Percentage of zero pixels: %s" % percentage)
    
    return percentage


def background_subtraction(img):
    """
        Gets the object to be tracked. Currently uses grab_cut
    """
        
    bg_mask = np.zeros(img.shape[:2],np.uint8)
    bg_model = np.zeros((1,65),np.float64) #internal to algo
    fg_model = np.zeros((1,65),np.float64) #internal to algo

    height, width, channels = img.shape

    #Center the area we are interested in cutting
    rect_x = int(width/4) 
    rect_y = int(height/4)
    rect_w = int(width*0.75) - rect_x
    rect_h = int(height*0.75) - rect_y

    rect = (rect_x, rect_y, rect_w, rect_h)
    cv2.grabCut(img,bg_mask,rect,bg_model,fg_model,1,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((bg_mask==2)|(bg_mask==0),0,1).astype('uint8')
    region = img*mask2[:,:,np.newaxis]

    #Find the largest contour
    region = find_largest_contour(region)

    #If grab cut failed, i.e. black pixels > 90%, just use the user defined region
    percent_zero_pixels = get_percentage_of_zero_pixels(region)

    if percent_zero_pixels > 0.85:
        ds.print_status(ds.FATAL_ERROR, "Grab cut failed. Using user selected region")
        cv2.imwrite("failed_grabcut.png", region)
        return img
    else:
        #Clean it up by finding the largest contour
        return region


def run(source = 0):
    """
        Main entry point of the script. 
    """
    #Get a video feed
    cap = cv2.VideoCapture(source)

    #Could not open camera device, quit
    if not cap.isOpened():
        ds.print_status(ds.FATAL_ERROR, "CAMERA COULD NOT BE OPENED. ABORTING")
        exit()
    
    #Wait for user to select a region
    ds.print_status(ds.INFO, "Press spacebar to start tracking")
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            ds.print_status(ds.FATAL_ERROR, "COULD NOT GET FRAME FROM CAMERA. ABORTING")
            exit()
        key = cv2.waitKey(10)
        if key == 32:
            break
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
    cv2.destroyWindow("Frame")
    
    #Resize the frame to a specific size
    resize_frame(frame, 2)

    #Allow the user to select a bounding box
    ref_points = rs.run(frame)
    roi = frame[ref_points[0][1]:ref_points[1][1],
                ref_points[0][0]:ref_points[1][0]]
        
    #Generate a model histogram to track
    is_tracking = False
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    if debug_mode:
        cv2.namedWindow("User Selected Region", cv2.WINDOW_NORMAL)
        cv2.imshow("User Selected Region", roi)

    #Background subtraction
    cut_image = background_subtraction(roi)
    if debug_mode:
        cv2.namedWindow("Object to be tracked", cv2.WINDOW_NORMAL)
        cv2.imshow("Object to be tracked", cut_image)

    
    #Convert to HSV space
    hsv_roi = cv2.cvtColor(cut_image, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.medianBlur(hsv_roi, 5)
    if debug_mode:
        cv2.namedWindow("HSV Region", cv2.WINDOW_NORMAL)
        cv2.imshow("HSV Region", hsv_roi)

    #Convert ROI to HSV
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)  

    #Compute adaptive mask values
    adaptive_mask_values = compute_mask(hsv_roi)
    mask = cv2.inRange(roi, 
                         np.array((adaptive_mask_values.low_h, adaptive_mask_values.low_s,adaptive_mask_values.low_v)), 
                         np.array((adaptive_mask_values.high_h,adaptive_mask_values.high_s,adaptive_mask_values.high_v)))
    if debug_mode:
        cv2.namedWindow("Adaptive Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Adaptive Mask", mask)

    #Calculate the histogram
    hue_bins = 180/4
    saturation_bins = 255/4
    if debug_mode:
        ds.print_status(ds.INFO, "Hue bins: %s" % str(hue_bins))
        ds.print_status(ds.INFO, "Saturation bins: %s" % str(saturation_bins))

    roi_hist = cv2.calcHist([roi], channels=[0,1], mask=mask, histSize=[hue_bins,saturation_bins], ranges=[0,180,0,255])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    if debug_mode:
        cv2.namedWindow("HS Histogram", cv2.WINDOW_NORMAL)
        cv2.imshow("HS Histogram", roi_hist)

    #END HISTOGRAM GENERATION
    #TODO MAKE THIS AUTO UPDATE

    #BEGIN TRACKING LOOP
    is_tracking = True
    track_window = (ref_points[0][0], ref_points[0][1], ref_points[1][0], ref_points[1][1])
    
    while is_tracking:
        #Get a frame
        (grabbed, frame) = cap.read()
        if debug_mode:
            cv2.namedWindow("Raw Feed", cv2.WINDOW_NORMAL)
            cv2.imshow("Raw Feed", frame)
        
        #Resize the frame to a specific size
        resize_frame(frame, 2)
        
        #Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if debug_mode:
            cv2.namedWindow("HSV Raw Feed", cv2.WINDOW_NORMAL)
            cv2.imshow("HSV Raw Feed", hsv_frame)

        #Calculate Backproject
        back_project = cv2.calcBackProject([hsv_frame], [0,1], roi_hist, ranges=[0,180,0,255], scale=1.0)
        if debug_mode:
            cv2.namedWindow("Back Projection", cv2.WINDOW_NORMAL)
            cv2.imshow("Back Projection", back_project)

        #Intialize camshift
        (ret, track_window) = cv2.CamShift(back_project, track_window, term_crit)
        x, y, w, h = track_window
        
        #Draw the tracks
        pts = np.int0(cv2.cv.BoxPoints(ret))
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        if debug_mode:
            cv2.namedWindow("Tracked", cv2.WINDOW_NORMAL)
            cv2.imshow("Tracked", frame)
        
        
        key = cv2.waitKey(10)
        if key == 27:
            break

    

if __name__ == "__main__":
    #Parse command line for video feed
    ds.print_status(ds.INFO, "Starting Hue/Saturation Camshift demo")

    ap = argparse.ArgumentParser()
    group = ap.add_argument_group()
    group.add_argument("-v", "--video",
            help="path to the (optional) video file")
    group.add_argument("-d", "--device",
            help="device id (i.e. 0)")
    group.add_argument("-r", "--release",
            help="start without debug flags")
    args = vars(ap.parse_args())

    if not args.get("video", False):
        ds.print_status(ds.INFO, ("No video file provided. Opening device id %s" % args["device"]))
        source = int(args["device"])
    else:
        ds.print_status(ds.INFO, ("Using video file: %s" % args["video"]))
        source = args["video"]
    debug_mode = bool(args["release"])
    while True:
        run(source)
        key = cv2.waitKey(0)
        if key == 27:
            break
