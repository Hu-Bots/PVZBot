from typing import Any
import pyautogui
from time import sleep
from detectors import *

available_sunlight = 50

class Plant():
    def __init__(self, name, position, cost, pixel_check, loaded_rgb, not_loaded_rgb, row_range=None, column_range=None):
        self.name = name
        self.position = position
        self.cost = cost
        self.pixel_check = pixel_check
        self.loaded_rgb = loaded_rgb
        self.not_loaded_rgb = not_loaded_rgb
        self.quantity_placed = 0
        self.row_range = row_range
        self.column_range = column_range

    
    def place(self, coordinates, hotkey=str):
        pyautogui.press(hotkey)
        sleep(.5)
        pyautogui.click(coordinates[0], coordinates[1])
        pyautogui.click(coordinates[0], coordinates[1])
        pyautogui.click(coordinates[0], coordinates[1])

    def is_loaded(self):
        #get loaded and unloaded colors
        lr, lg, lb = self.loaded_rgb
        ur, ug, ub = self.not_loaded_rgb

        #establish mid point
        mid_r = round((lr+ur)/2)
        mid_g = round((lg+ug)/2)
        mid_b = round((lb+ub)/2)

        #get current rgb
        img = pyautogui.screenshot("train.png", (self.pixel_check[0],self.pixel_check[1],1,1))
        current_r, current_g, current_b = img.getpixel((0,0))

        #determine state
        if current_r > mid_r and current_g > mid_g and current_b > mid_b:
            return True
        else:
            return False



class Zombie():
    def __init__(self, query):
        self.query = query

    def locate(self, min_matches):        
        position = SIFT_method(
            query_img  = self.query,
            min_matches = min_matches
        )
        return position


class GameMap():
    def __init__(self):

        #0 = empty, sf = sunflower, ps = peashooter
        self.grid = [
            [0,0,0,0,0,0,0,0,0], #1
            [0,0,0,0,0,0,0,0,0], #2
            [0,0,0,0,0,0,0,0,0], #3
            [0,0,0,0,0,0,0,0,0], #4
            [0,0,0,0,0,0,0,0,0], #5
        ]

        self.tile_shape  = 133,156 #w,h
        self.topleft = 550,178 #x, y
        self.shape = 1195,815

    def check_collision(self, coordinates):
        x = coordinates[0]
        y = coordinates[1]
        for row_idx, row in enumerate(self.grid):
            #check y collision
            bounding_y_start = row_idx * self.tile_shape[1]
            bounding_y_end   = (row_idx+1) * self.tile_shape[1]

            if y >= bounding_y_start and y <= bounding_y_end:
                y_collision = True

                for col_idx, column in enumerate(row):
                    #check x collision
                    bounding_x_start = col_idx * self.tile_shape[0]
                    bounding_x_end   = (col_idx+1) * self.tile_shape[0]

                    if x >= bounding_x_start and x <= bounding_x_end:
                        x_collision = True

                        return [row_idx, col_idx]
    
    def is_defended(self, row):
        row = self.grid[row]
        for column in row:
            if column == "peashooter":
                return True
        return False

    def index_to_coordinate(self, index):
        row = index[0]
        column = index[1]

        x = (column*self.tile_shape[0])+(round(self.tile_shape[0]/2))+self.topleft[0]
        y = (row*self.tile_shape[1])+(round(self.tile_shape[1]/2))+self.topleft[1]

        return [x,y]

    def determine_best_plant_posision(self, row_range, column_range, plant=str):
        for column_idx in range(column_range[0], column_range[1]+1):
            #col 0
            for row_idx in range(row_range[0], row_range[1]+1):
                if self.grid[row_idx][column_idx] == 0:
                    print(f'Best placing: {row_idx}, {column_idx}')
                    return row_idx, column_idx



def gather_sunrays():
    pos = matchtemplate_method(
        query      = sunray_query,
        confidence = .7
    )

    if type(pos) is tuple:
        #add gap
        x, y =  pos
        x += game_map.topleft[0]
        y += game_map.topleft[1]

        pyautogui.click(x, y+40)
        sleep(.5)
        return True
    else:
        return False

def take_screenshot():
    return pyautogui.screenshot("train.png", (game_map.topleft[0], game_map.topleft[1], game_map.shape[0], game_map.shape[1]))

#PLANTS
peashooter = Plant(
    name           = 'peashooter',
    position       = (132,89),
    cost           = 100,
    pixel_check    = [120,47],
    loaded_rgb     = (212, 253, 49),
    not_loaded_rgb = (62, 77, 12),
    row_range      = [0,4],
    column_range   = [2,8]
)

sunflower = Plant(
    name           = 'sunflower',
    position       = (128,172),
    cost           = 50,
    pixel_check    = (128,172),
    loaded_rgb     = (206,183,5),
    not_loaded_rgb = (101,92,0),
    row_range      = [0,4],
    column_range   = [0,1]
)

potato_mine = Plant(
    name           = 'potato_mine',
    position       = (108,532),
    cost           = 25,
    pixel_check    = (108,532),
    loaded_rgb     = (244,20,14),
    not_loaded_rgb = (72,3,1),
    row_range      = [0,4],
    column_range   = [3,3]
)

#ZOMBIES
normal_zombie = Zombie(
    query   = cv.imread("zombie.png", 0)
)

#MAP
game_map = GameMap()
sunray_query = cv.imread("sunray.png", 0)

#MAIN
sleep(3)

while True:
    #GATHER SUN RAYS
    take_screenshot()
    gathering = gather_sunrays()
    if gathering == True:
        available_sunlight += 25
        continue

    #IF NONE AVAILABLE, CHECK FOR ZOMBIES
    take_screenshot()
    zombie_scan = normal_zombie.locate(min_matches=10)

    if type(zombie_scan) is list:
        zombie_index = game_map.check_collision(zombie_scan)
        if zombie_index is None:
            continue
        zombie_row = zombie_index[0]
        zombie_column = zombie_index[1]
        print(f'Zombie found in grid [{zombie_row, zombie_column}]')
    
        #IF ZOMBIE IN SIGHT, CHECK IF ROW IS DEFENDED
        row_is_defended = game_map.is_defended(zombie_row)

        #IF NOT DEFENDED, PLACE MINE IN THE THIRD COLUMN OF THAT ROW
        if row_is_defended == False:
            index = [zombie_row, potato_mine.column_range[0]]
            coordinates = game_map.index_to_coordinate(index)
            if potato_mine.is_loaded() == True:
                potato_mine.place(coordinates, '5')
                available_sunlight-=potato_mine.cost
                potato_mine.quantity_placed+=1

    #PLACE PEASHOOTER IF ENOUGH SUNLIGHT
    if available_sunlight >= peashooter.cost:
        place_idx = game_map.determine_best_plant_posision(
            row_range   = peashooter.row_range,
            column_range= peashooter.column_range,
            plant       = peashooter.name
        )
        if place_idx is not None and peashooter.is_loaded() == True:
            row, column = place_idx
            coordinates = game_map.index_to_coordinate((row, column))
            peashooter.place(coordinates,'1')
            available_sunlight-=peashooter.cost
            game_map.grid[row][column] = peashooter.name
            print('placed peashooter')

    #IF NOT, PLACE SUNFLOWER
    if available_sunlight >= sunflower.cost:
        place_idx = game_map.determine_best_plant_posision(
            row_range   = sunflower.row_range,
            column_range= sunflower.column_range,
            plant       = sunflower.name
        )
        if place_idx is not None and sunflower.is_loaded() == True:
            row, column = place_idx                
            coordinates = game_map.index_to_coordinate((row,column))
            sunflower.place(coordinates,'2')
            available_sunlight-=sunflower.cost
            game_map.grid[row][column] = sunflower.name
            print('placed sunflower')