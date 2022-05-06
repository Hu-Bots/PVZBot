import cv2 as cv
from cv2 import imshow
import pyautogui
import numpy as np

pyautogui.PAUSE = False

def matchtemplate_method(query, train="train.png", confidence = .8):
    """
    @query = img object
    @train_path  = string
    @confidence = float: 0-1
    @train_shape = tuple with 4 digits
    """

    #template and dimensions
    template = query
    template_w, template_h = template.shape[::-1]

    if type(train) is str:
        image = cv.imread(train, 0)
    else:
        image = train

    result = cv.matchTemplate(
        image  = image,
        templ  = template,
        method = cv.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    #threshold
    if max_val >= confidence:
        return max_loc


def SIFT_method(query_img, train_path = "train.png", min_matches=10, confidence = 0.75, show_query_keypoints=False):
    """
    @param query_img    = img object
    @param train_path   = string
    @param min_matches  = int
    @param position     = boo
    @param confidence   = float 0-1
    @param show_keypoints = boolean, returns function
    """

    img1 = query_img
    img2 = cv.imread(train_path,0) # trainImage

    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # matching algorythm
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2) #(tuple with matches)

    if show_query_keypoints:
        query_kps = cv.drawKeypoints(img1, kp1, img1)
        cv.imwrite("query keypoints.png", query_kps)
        print("Query keypoints: ", len(kp1), ". Image created.")
        return


    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < confidence*n.distance:
            good.append(m)

    if len(good)>min_matches:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M) # 4 coordinates

        coords1_x, coords1_y = dst[0][0]
        coords2_x, coords2_y = dst[1][0]
        coords3_x, coords3_y = dst[2][0]
        coords4_x, coords4_y = dst[3][0]

        center_x = round((coords1_x+coords2_x+coords3_x+coords4_x)/4)
        center_y = round((coords1_y+coords2_y+coords3_y+coords4_y)/4)

        return [center_x, center_y]

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), min_matches) )
        matchesMask = None
        return None