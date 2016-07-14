import cv2
import debug_status as ds

def run(image):
    """
        Main Entry point for the ROI selector
    """
    display_img = image.copy()
    draw_img = image.copy()

    window_name = "Select object to be tracked in this window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, draw_img)

    run.first_point = {}
    run.second_point = {}

    run.mouse_down = False

    #Closure to handle mouse callbacks
    def mouse_callback(event, x, y, flags, param):
        """
            Callback for the detected mouse event
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            run.mouse_down = True
            run.first_point = (x,y)
        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down:
            run.mouse_down = False
            run.second_point = (x, y)
            ds.print_status(ds.INFO, "Got ROI")
        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down:
            dis = draw_img.copy()
            cv2.rectangle(dis, run.first_point, (x, y), (255,255,255), 3)
            cv2.imshow(window_name, dis)

    ds.print_status(ds.INFO, "Select object to be tracked with the mouse")
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        key = cv2.waitKey(30)
        if key == 32:
            cv2.destroyWindow(window_name)
            return (run.first_point, run.second_point)

def check_points(first, second):
    """
     Ensures a set of points are opposit on the plane
     return: [max_x, min_x, max_y, min_y]
    """
    ret_point = [0,0,0,0]
    #Gets the max and min x coordinate
    if first[0] < second[0]:
        ret_point[0] = second[0]
        ret_point[1] = first[0]
    else:
        ret_point[0] = first[0]
        ret_point[1] = second[0]
        
    #Gets the max and min y coordinate
    if first[1] < second[1]:
        ret_point[2] = second[1]
        ret_point[3] = first[1]
    else:
        ret_point[2] = first[1]
        ret_point[3] = second[1]
    return ret_point


