from detectors import matchtemplate_method, SIFT_method
import cv2
from time import sleep
import pyautogui

game_area = (543,171,1233,827) #x, y, w, h

#load templates
sunray_img = cv2.imread("imgs/sunray.png", 0)

def gather_sunrays():
    """Gather all available sunrays."""

    sunrays_gathered = 0
    pyautogui.screenshot("train.png", game_area)
    train_img = cv2.imread("train.png", 0)

    while True:
        result = matchtemplate_method(sunray_img, train_img)

        if type(result) == tuple:
            result_x, result_y = result

            #cover sunray so it isn't detected again
            sunray_imgW, sunray_imgH = sunray_img.shape[::-1]
            cv2.rectangle(train_img, (result_x, result_y), (result_x+sunray_imgW, result_y+sunray_imgH), (255,0,0), -1)

            #get coordinates in screen
            screen_x = result_x + game_area[0]
            screen_y = result_y + game_area[1]

            #click on the lower part of the sunray
            pyautogui.click(screen_x, screen_y+sunray_imgH)
            sunrays_gathered += 25
            sleep(.5) #gives the game window time to update.
        else:
            return sunrays_gathered

def main():
    current_sunrays = 50
    while True:
        
        current_sunrays+=gather_sunrays()
        print(current_sunrays)

main()