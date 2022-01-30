import cv2 as cv
import pyautogui
import numpy as np
from matplotlib import pyplot as plt

pyautogui.PAUSE = False

def matchtemplate_method(query, train_path="train.png", show=False, position = True, confidence = .8):
    """
    @param query = img object
    @train_path  = string
    @show        = boo
    @position    = boo
    """

    #template and dimensions
    template = query
    template_w, template_h = template.shape[::-1]

    image = cv.imread(train_path, 0)

    result = cv.matchTemplate(
        image  = image,
        templ  = template,
        method = cv.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    #threshold
    if max_val >= confidence:

        if position == True:
            return max_loc

        if show == True:

            image = cv.rectangle(
                img = image,
                pt1 = max_loc,
                pt2 = (
                    max_loc[0] + template_w, # = pt2 x 
                    max_loc[1] + template_h # = pt2 y
                ),
                color = (0,0,255),
                thickness = -1 #fill the rectangle
            )
            cv.imshow("Match Template Result", image)
            cv.waitKey()


def SIFT_method(query_img, train_path = "train.png", min_matches=10, position=True, show=False, confidence = 0.75):
    """
    @param query_img    = img object
    @param train_path   = string
    @param min_matches  = int
    @param position     = boo
    @param confidence   = float
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

        if position == True:
            return [center_x, center_y]

        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), min_matches) )
        matchesMask = None
        return False

    draw_params = dict(
                    matchColor       = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask      = matchesMask, # draw only inliers
                    flags            = 2
                    )

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    if show == True:
        plt.imshow(img3, 'gray'), plt.show()