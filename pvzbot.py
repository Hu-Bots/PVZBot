from detectors import matchtemplate_method, SIFT_method
import cv2
from time import sleep
import pyautogui

game_area = (543,171,1233,827) #x, y, w, h

#load templates
sunray_img = cv2.imread("imgs/sunray.png", 0)
zombie_img = cv2.imread("imgs/zombie.png", 0)

game_grid = [
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0]
]
grid_topleft = game_area[0], game_area[1]
grid_incrementX = 135
grid_incrementY = 165

def zombieDetector():
    pyautogui.screenshot("train.png", game_area)
    result = SIFT_method(zombie_img)

    if type(result) is list:
        screen_x = result[0] + game_area[0]
        screen_y = result[1] + game_area[1]

        if 0>screen_x>1920 or 0>screen_y>1080:
            print(f"Zombie was found, but we were unable to specify the location: {screen_x}, {screen_y}")
            return
        else:
            indexes = coordsToIndex((screen_x, screen_y))
            if indexes is None:
                return
            else:
                row, column = indexes
                print(f"Zombie incoming at {row},{column}.")
                return row, column

def coordsToIndex(coords):
    x, y = coords
    for row_num, row in enumerate(game_grid):
        for column_num , column in enumerate(row):
            start_x = grid_topleft[0] + (grid_incrementX*column_num)
            start_y = grid_topleft[1] + (grid_incrementY*row_num)

            end_x = start_x + grid_incrementX
            end_y = start_y + grid_incrementY

            if start_x <= x <= end_x and start_y <= y <= end_y:
                return row_num, column_num


def drawGrid():
    grid_img = cv2.imread("imgs/grid_sample.png")
    for row_num, row in enumerate(game_grid):
        for column_num , column in enumerate(row):
            start_x = grid_topleft[0] + (grid_incrementX*column_num)
            start_y = grid_topleft[1] + (grid_incrementY*row_num)

            end_x = start_x + grid_incrementX
            end_y = start_y + grid_incrementY

            cv2.rectangle(grid_img, (start_x, start_y), (end_x,end_y), (255,0,0), 3)

    cv2.imshow("grid", grid_img)
    cv2.waitKey()

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
        else:
            return sunrays_gathered

def main():
    current_sunrays = 50
    while True:
        
        current_sunrays+=gather_sunrays()
        print(current_sunrays)
        sleep(.5) #gives the game window time to update.

        zombieDetector()

main()