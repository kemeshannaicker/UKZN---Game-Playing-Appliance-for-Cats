#MOUSE HUNTER V1.4 -By Kemeshan Naicker

# CHANGELOG: added game time,restart option, and class for adding buttons

### IMPORTS SECTION ###

import cv2
import pygame
from pygame.locals import *
import numpy as np
from math import sqrt
import random
import pickle
import time
### END OF IMPORTS SECTION ###

### DEFINITIONS SECTION ###

# GAME DEFINITIONS
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
DIAGONAL_VEL = 10
AXIS_VEL = int(sqrt(2*DIAGONAL_VEL*DIAGONAL_VEL))
COL_DIR_SCALAR = 3
TIME_PRD = 5
SPRITE_WIDTH = 60
SPRITE_HEIGHT = SPRITE_WIDTH
BUTTON_WIDTH = 50
BUTTON_HEIGHT = 50
COL_RADIUS = 30
BUTTON_COL_RADIUS = 8
BG_COLOR = 'white'

# IMAGE PROCESSING DEFINITIONS
BUFFER = 50                               #buffer of pixels around the screen
OBJ_THRESHOLD = 30                      #binary threshold for detecting dynamic objects
BUTTON_Y_PROP = 0.5
BUTTON_X_PROP = 0.2
calibration_message = 'Calibrating. Please darken the room if possible then press c to proceed.'
warning_message = 'Once you proceed, the screen will go blank for a few seconds while adjusting. Please vacate the area in front of the screen.'
font = cv2.FONT_HERSHEY_PLAIN
### END OF DEFINITIONS SECTION ###

### LOOKUP TABELS SECTION ###

# GAME LOOKUP TABLES
zero_biased_dir = [-4,-3,-2,-1,1,2,3,4,0,0,0,0,0]
all_directions = [-4,-3,-2,-1,1,2,3,4]      #array of directions for enemy to travel
pos_x = [1,2,-4]                          #positive x directions
pos_y = [-2,-3,-4]                        #negative x directions
neg_x = [-1,-2,4]                         #positive y directions
neg_y = [2,3,4]                           #negative y directions
animation_cds = [1,1,1]
### END OF LOOKUP TABLES SECTION ###

### GLOBAL VARIABLES SECTION ###

# GLOBAL GAME VARIABLES
score = 0
start_menu = True
game_length = 20
clock = pygame.time.Clock()

# GLOBAL IMAGE PROCESSING VARIABLES
background = None
button_background = None
screen_accum_weight = 0.5
btn_accum_weight = 0.5
frames_elapsed = 0
mouse_x = 0
mouse_y = 0
foundObject = False
tx = -100
ty = -100
bx = -100
by = -100
lx = -100
ly = -100
rx = -100
ry = -100

# GLOBAL TROUBLESHOOTING VARIABLES
troubleshoot = True 
### END OF GLOBAL VARIABLES SECTION ###

### FUNCTION DEFINITIONS SECTION

# function to calculate distance between points in 2D and 3D space
def distance(p,q):
    return sqrt(sum([(a-b)**2 for a,b in zip(p, q)]))


def undistort(img, mtx, dist, form=0):
    
    h,w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), form, (w,h))
    
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    #crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def get_processed_frame():
    _, frame = cap.read()
    undistorted_frame = undistort(frame,mtx,dist)
    grey_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
    return grey_frame

def crop_frame_to_screen(frame):
    cropped = frame[roi_topY:roi_bottomY,roi_leftX:roi_rightX]
    return cropped

def crop_frame_to_btns(frame):
    cropped = frame[screen_top_y:screen_top_y+int(screen_height*BUTTON_Y_PROP),screen_left_x:screen_left_x+int(screen_width*BUTTON_X_PROP)]
    return cropped

# function to calculate the accumulated average of an image passed in given a new frame and a previous screen_accum_weight
def calc_screen_accum_avg(frame, screen_accum_weight):
    global background
    
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame, background, screen_accum_weight)
    return None
    
def calc_btn_accum_avg(roi, btn_accum_weight):
    global button_background
    
    if button_background is None:
        button_background = roi.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(roi, button_background, btn_accum_weight)
    return None

# find the largest object in the foreground
def findObject(frame, background, threshold=OBJ_THRESHOLD, minSize=300, blur=True):
    global foundObject, tx, ty, bx, by, rx, ry, lx, ly
    if blur:
        frame = cv2.GaussianBlur(frame, (7,7), 0)
    diff = cv2.absdiff(background.astype('uint8'), frame)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    if len(contours) == 0:
        foundObject = False
        tx = -100
        return None
        
    else:
        
        largest_obj = max(contours, key=cv2.contourArea)
            
        if cv2.contourArea(largest_obj) < minSize:              
            foundObject = False
            tx = -100
            ty = -100
            bx = -100
            by = -100
            lx = -100
            ly = -100
            rx = -100
            ry = -100
            return None
    
    foundObject = True
    return (thresholded, largest_obj)   
    
# function to find pointed fingers
def getOuterPoints(largest_obj):
    global tx, ty, bx, by, lx, ly, rx, ry
    conv_hull = cv2.convexHull(largest_obj)
    obj_top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    obj_bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    obj_left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    obj_right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    tx = obj_top[0]
    ty = obj_top[1]
    bx = obj_bottom[0]
    by = obj_bottom[1]
    lx = obj_left[0]
    ly = obj_left[1]
    rx = obj_right[0]
    ry = obj_right[1]

#function to convert from remap a coordinate to an image of different size
def remap(x,y,buffer=0):
    remapped = (buffer+int((x/SCREEN_WIDTH)*(screen_right_x-screen_left_x)),int((y/SCREEN_HEIGHT)*(screen_bottom_y-screen_top_y))) 
    return remapped

### CLASS DEFINITIONS SECTIONS

class Enemy():
    def __init__(self, x, y):
        #lists for sprite images:
        self.images_right = []                                                      
        self.images_left = []
        self.images_up = []
        self.images_down = []
        self.index = 0                                                              #index for animation state
        for num in range (0,3):                                                     #load sprite images
            img_right = pygame.image.load(f'assets/sprites/rat_right_{num}.png')
            img_right = pygame.transform.scale(img_right, (SPRITE_WIDTH,SPRITE_HEIGHT))
            self.images_right.append(img_right)
            img_left = pygame.image.load(f'assets/sprites/rat_left_{num}.png')
            img_left = pygame.transform.scale(img_left, (SPRITE_WIDTH,SPRITE_HEIGHT))
            self.images_left.append(img_left)
            img_up = pygame.image.load(f'assets/sprites/rat_up_{num}.png')
            img_up = pygame.transform.scale(img_up, (SPRITE_WIDTH,SPRITE_HEIGHT))
            self.images_up.append(img_up)
            img_down = pygame.image.load(f'assets/sprites/rat_down_{num}.png')
            img_down = pygame.transform.scale(img_down, (SPRITE_WIDTH,SPRITE_HEIGHT))
            self.images_down.append(img_down)
        self.image = self.images_right[self.index]                                  #assign initial image
        self.rect = self.image.get_rect()                                           #get co-ordinates of rectangle containing sprite
        self.rect.x = x                                                             #get sprite x and y co-ordinates
        self.rect.y = y
        self.centre = remap(self.rect.x+SPRITE_WIDTH//2, self.rect.y+SPRITE_HEIGHT//2, BUFFER)   #get centre of sprite
        self.direction = 0                                                          #initialise direction of movement
        self.vel = 1                                                                #initialise velocity scalar
        self.score_cd = 0                                                           #cooldown before score can be updated (initially zero                                                          
        self.direction_cd = 5                                                       #cooldown before direction can change
        self.animation_cd = 1                                                       #initial cooldown for animation transitions
        self.cur_animation_cd = 1                                                   #cooldown for animation transitions updated based on current velocity
        
    def reset(self):
        global score
        gamescreen.blit(bg_img, (0,0))
        score = 0
        self.index = 0
        self.image = self.images_right[self.index]                                  #assign initial image
        self.rect = self.image.get_rect()                                           #get co-ordinates of rectangle containing sprite
        self.rect.x = random.randint(1,12)*100                                                             #get sprite x and y co-ordinates
        self.rect.y = random.randint(1,7)*100
        self.centre = remap(self.rect.x+SPRITE_WIDTH//2, self.rect.y+SPRITE_HEIGHT//2, BUFFER)   #get centre of sprite
        self.direction = 0                                                          #initialise direction of movement
        self.vel = 1                                                                #initialise velocity scalar
        self.score_cd = 0                                                           #cooldown before score can be updated (initially zero                                                          
        self.direction_cd = 5                                                       #cooldown before direction can change
        self.animation_cd = 1                                                       #initial cooldown for animation transitions
        self.cur_animation_cd = 1
        
    def update(self, tx, ty, lx, ly, rx, ry):
        global score, mouse_x, mouse_y, foundObject
        dx = 0
        dy = 0
        gamescreen.blit(bg_square, (self.rect.x,self.rect.y))
        self.direction_cd -= 1                                                      #decrement cd on direction change
        if self.score_cd != 0:                                                      #decrement score cd unless it is already 0
            self.score_cd -= 1
        if self.direction_cd == 0:                                                  #if direction cd is 0
            self.direction = zero_biased_dir[random.randint(0,len(zero_biased_dir)-1)]                                   #get new randomized direction
            self.vel = random.randint(2,4)                                          #get new randomized velocity
            self.cur_animation_cd = animation_cds[self.vel-2]                       #update animation transition rate based on new velocity
            interval = random.randint(1,3)                                          #get randomised time period for traveling in this new direction
            self.direction_cd = interval*TIME_PRD
            
        distances = [distance(self.centre,(tx,ty)), distance(self.centre,(lx,ly)), distance(self.centre,(rx,ry))]                                    #check distance between object and cat
        dist = min(distances)
        
        if foundObject and (dist < COL_RADIUS):                                                       #if distance is smaller than the defined collision radius:
            if self.score_cd == 0:                                                  #update the score, play sound
                
                ###########################################################
#                 if troubleshoot:
#                     cv2.imwrite(f'troubleshooting/collisions/contact{file_index}.jpg', frame_copy)
#                     file_index += 1
                ###########################################################
                
                squeak_sound.play(0)
                self.score_cd = 5
                score += 1
                cur_direction = self.direction
                while self.direction == cur_direction:                              #change direction
                    self.direction = all_directions[random.randint(0,7)]
            self.vel = 5                                                            #set velocity to maximum
            self.direction_cd = 5*TIME_PRD                                          #set time period of travel to custom maximum

        
        #define directional movement and animations
        if self.direction == 1:         #right
            dx = AXIS_VEL															#update proposed changes in x-y coordinate based on direction
            dy = 0																	
            self.animation_cd -= 1													#decrement animation_cd
            if self.animation_cd == 0:												#if enough time has passed
                self.animation_cd = self.cur_animation_cd							#set new animation_cd based on current velocity
                self.index += 1														#use next sprite image to create animation effect
                if self.index >= len(self.images_right):							#wrap around animation lookup table
                    self.index = 0
                self.image = self.images_right[self.index]
                
        elif self.direction == 2:       #up+right
            dx = DIAGONAL_VEL
            dy = -DIAGONAL_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                self.image = self.images_right[self.index]
                
        elif self.direction == 3:       #up
            dx = 0
            dy = -AXIS_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_up):
                    self.index = 0
                self.image = self.images_up[self.index]
                
        elif self.direction == 4:       #up+left
            dx = -DIAGONAL_VEL
            dy = -DIAGONAL_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -1:      #left
            dx = -AXIS_VEL
            dy = 0
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -2:      #down+left
            dx = -DIAGONAL_VEL
            dy = DIAGONAL_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -3:      #down
            dx = 0
            dy = AXIS_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_down):
                    self.index = 0
                self.image = self.images_down[self.index]
                
        elif self.direction == -4:      #down+right
            dx = DIAGONAL_VEL
            dy = DIAGONAL_VEL
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                self.image = self.images_right[self.index]

        #check for collisions and update position
        if (self.rect.x + dx*self.vel) < 0:                 						#check for collision with left of screen
            self.direction = pos_x[random.randint(0,2)]     
            self.direction_cd = COL_DIR_SCALAR*TIME_PRD
        elif (self.rect.x + dx*self.vel) > SCREEN_WIDTH:    						#check for collision with right of screen
            self.direction = neg_x[random.randint(0,2)]
            self.direction_cd = COL_DIR_SCALAR*TIME_PRD
        else:
            self.rect.x += dx*self.vel                      
        if (self.rect.y + dy*self.vel) < 0:                 						#check for collision with top of screen
            self.direction = pos_y[random.randint(0,2)]
            self.direction_cd = COL_DIR_SCALAR*TIME_PRD
        elif (self.rect.y + dy*self.vel) > SCREEN_HEIGHT:   						#check for collision with bottom of screen
            self.direction = neg_y[random.randint(0,2)]
            self.direction_cd = COL_DIR_SCALAR*TIME_PRD
        else:
            self.rect.y += dy*self.vel

        self.centre = remap(self.rect.x+SPRITE_WIDTH//2, self.rect.y+SPRITE_HEIGHT//2, BUFFER)	#calculate new_centre
        mouse_x = self.centre[0]
        mouse_y = self.centre[1]
        gamescreen.blit(self.image, self.rect)										#blit object to screen
        
class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x                                                             #get sprite x and y co-ordinates
        self.rect.y = y
        self.centre = remap(self.rect.x+BUTTON_WIDTH//2,self.rect.y+BUTTON_HEIGHT//2)
        self.press_cd = 0
        
    def draw(self):
        gamescreen.blit(self.image, self.rect)
        
    def is_pressed(self):
        global foundObject
        distances = [distance(self.centre,(tx,ty)),distance(self.centre,(bx,by)), distance(self.centre,(lx,ly)), distance(self.centre,(rx,ry))]                                    #check distance between button and hand
        dist = min(distances)

        if foundObject and (dist < BUTTON_COL_RADIUS):
            if self.press_cd == 0:
                self.press_cd = 5
                return True
            else:
                self.press_cd -= 1
        
        return False
        
### INITIALISATION SECTION
        
# IMAGE PROCESSING INITIALISATION
# load camera parameters
file = open('camera_params.txt', 'rb')
dict = pickle.load(file)
file.close()

mtx = dict['matrix']
dist = dict['distortion']

# create blank image for screen detection
white = np.ones(shape=(720, 1280, 3),dtype=np.uint8)*255
cv2.putText(white,text=calibration_message,org=(150,320),fontFace = font,fontScale=1.5,color=(50,50,50),thickness=3)
cv2.putText(white,text=warning_message,org=(100,380),fontFace = font,fontScale=1,color=(50,50,50),thickness=2)

# create video capture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS,20)

# INITIALISATION OF ROI
cv2.namedWindow('White Screen', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('White Screen', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True: 																		#capture frame of blank screen
    cv2.imshow('White Screen', white)
    ret, frame = cap.read()
    if cv2.waitKey(1)== ord('c'):
        
        #########################################################
        if troubleshoot:
            cv2.imwrite('troubleshooting/screen.jpg',frame)
        #########################################################
            
        break    
cv2.destroyAllWindows()

calib_img = undistort(frame, mtx, dist)                               				#undistort the image                      

#################################################################
if troubleshoot:
    cv2.imwrite('troubleshooting/screen_undistorted.jpg', calib_img)
#################################################################
    
grey_img = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)                            	#convert to greyscale
ret, thresh = cv2.threshold(grey_img,110,255,cv2.THRESH_BINARY)                    	#apply binary threshold

#################################################################
if troubleshoot:
    cv2.imwrite('troubleshooting/screen_thresh.jpg', thresh)
#################################################################
    
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   # find contours
screen = max(contours, key = cv2.contourArea)                                            # find contour with largest area -> screen
screen_left_x,screen_top_y,screen_width,screen_height = cv2.boundingRect(screen)                                                       # find bounding rectangle of screen
screen_right_x = screen_left_x + screen_width
screen_bottom_y = screen_top_y + screen_height

#################################################################
if troubleshoot:
    cv2.rectangle(calib_img, (screen_left_x,screen_top_y), (screen_right_x,screen_bottom_y), (255,0,0), 2)
    cv2.imwrite('troubleshooting/screen_bounded.jpg',calib_img)
#################################################################

roi_leftX = screen_left_x-BUFFER																#save ROI co-ordinates for further use
roi_rightX = screen_right_x+BUFFER
roi_topY = screen_top_y
roi_bottomY = screen_bottom_y+BUFFER*2

# INITIALISATION OF BACKGROUND MASK
white = np.ones((720,1280,3),dtype='uint8')*255
cv2.namedWindow('Background Screen', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Background Screen', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
while frames_elapsed < 60:															#calculate the average background over 60 frames
    frames_elapsed += 1
    cv2.imshow('Background Screen', white)
    cv2.waitKey(1)
    grey = get_processed_frame()
    grey_screen = crop_frame_to_screen(grey)
    grey_screen = cv2.GaussianBlur(grey_screen, (7,7), 0)
    calc_screen_accum_avg(grey_screen, screen_accum_weight)
    grey_btn_roi = crop_frame_to_btns(grey)
    calc_btn_accum_avg(grey_btn_roi, btn_accum_weight)    
cv2.destroyAllWindows()

# GAME INITIALISATION
#initialize pygame
pygame.init()

#load images
bg_img = pygame.image.load(f'assets/backgrounds/{BG_COLOR}.png')
bg_square = pygame.image.load(f'assets/sprites/{BG_COLOR}_square.png')
bg_square = pygame.transform.scale(bg_square, (SPRITE_WIDTH,SPRITE_HEIGHT))
start_menu_img = pygame.image.load(f'assets/backgrounds/start_menu.png')
restart_button_img = pygame.image.load('assets/buttons/restart_button.png')
restart_button_img = pygame.transform.scale(restart_button_img, (BUTTON_WIDTH,BUTTON_HEIGHT))
exit_button_img = pygame.image.load('assets/buttons/exit_button.png')
exit_button_img = pygame.transform.scale(exit_button_img, (BUTTON_WIDTH,BUTTON_HEIGHT))
plus_button_img = pygame.image.load('assets/buttons/plus_button.png')
plus_button_img = pygame.transform.scale(plus_button_img, (BUTTON_WIDTH,BUTTON_HEIGHT))
minus_button_img = pygame.image.load('assets/buttons/minus_button.png')
minus_button_img = pygame.transform.scale(minus_button_img, (BUTTON_WIDTH,BUTTON_HEIGHT))
play_button_img = pygame.image.load('assets/buttons/play_button.png')
play_button_img = pygame.transform.scale(play_button_img, (BUTTON_WIDTH,BUTTON_HEIGHT))

#load fonts
myfont = pygame.font.SysFont('Arial',30)

#load sounds
pygame.mixer.init()
squeak_sound = pygame.mixer.Sound('assets/sounds/squeak1.wav')
pygame.mixer.music.load('assets/sounds/background_music.wav')

#create game object instances
enemy = Enemy(100,100)
restart_button = Button(100,50,restart_button_img)
exit_button = Button(100,120,exit_button_img)
plus_button = Button(100,50,plus_button_img)
plusx = plus_button.centre[0]
plusy = plus_button.centre[1]
minus_button = Button(100,150,minus_button_img)
play_button = Button(100,220,play_button_img)

#create game window and caption
gamescreen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT), pygame.NOFRAME)       
pygame.display.set_caption('Mouse Hunter')
pygame.mixer.music.play(-1)

# MAIN LOOP SECTION
run = True
while run:
    
    ######################################################
    if troubleshoot:
        _, marker_frame = cap.read()
        marker_frame = undistort(marker_frame,mtx,dist)
        marker_frame = crop_frame_to_screen(marker_frame)
    ######################################################
    
    if start_menu:
        gamescreen.blit(start_menu_img, (0,0))
        game_lengthText = myfont.render(str(game_length)+" seconds",3,(0,0,0))
        gamescreen.blit(game_lengthText, (50,110))
        plus_button.draw()
        minus_button.draw()
        play_button.draw()
        pygame.display.update()
        frame = get_processed_frame()
        roi = crop_frame_to_btns(frame)
        found_hand = findObject(roi,threshold=50,minSize=200,background=button_background,blur=False)
        if found_hand is not None:
            thresholded, largest_obj = found_hand
            getOuterPoints(largest_obj)
            
            if plus_button.is_pressed():
                game_length += 10
            elif minus_button.is_pressed() and game_length > 10:
                game_length -= 10
            elif play_button.is_pressed():
                start_menu = False
                start_time = time.time()
                time.process_time()
                gamescreen.blit(bg_img, (0,0))
                pygame.mixer.music.load('assets/sounds/ingame_background.wav')
                pygame.mixer.music.play(-1)
                
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == ord('x'):
                    run = False
                elif event.key == ord('b'):
                    start_menu = False
                    start_time = time.time()
                    time.process_time()
                    gamescreen.blit(bg_img, (0,0))
                    pygame.mixer.music.load('assets/sounds/ingame_background.wav')
                    pygame.mixer.music.play(-1)
                elif event.key == pygame.K_EQUALS:
                    game_length += 10
                elif event.key == pygame.K_MINUS and game_length > 10:
                    game_length -= 10
                    
        ###############################################################            
        if troubleshoot:
            cv2.circle(marker_frame, (tx+BUFFER,ty), 3, (255,0,0), -1)                 #uncomment to mark top-most point of object
            cv2.circle(marker_frame, (tx+BUFFER,ty), BUTTON_COL_RADIUS, (255,0,0), 3)         #uncomment to mark radius of collision ^
            cv2.circle(marker_frame, (bx+BUFFER,by), 3, (0,255,255), -1)                 #uncomment to mark top-most point of object
            cv2.circle(marker_frame, (bx+BUFFER,by), BUTTON_COL_RADIUS, (0,255,255), 3)
            cv2.circle(marker_frame, (lx+BUFFER,ly), 3, (0,255,0), -1)                 #uncomment to mark left-most point of object
            cv2.circle(marker_frame, (lx+BUFFER,ly), BUTTON_COL_RADIUS, (0,255,0), 3)         #uncomment to mark radius of collision ^
            cv2.circle(marker_frame, (rx+BUFFER,ry), 3, (0,0,255), -1)                 #uncomment to mark right-most point of object
            cv2.circle(marker_frame, (rx+BUFFER,ry), BUTTON_COL_RADIUS, (0,0,255), 3)
            cv2.circle(marker_frame, (plus_button.centre[0]+BUFFER,plus_button.centre[1]), 5, (0,128,255), -1)
            cv2.circle(marker_frame, (minus_button.centre[0]+BUFFER,minus_button.centre[1]), 5, (0,128,255), -1)
            cv2.circle(marker_frame, (play_button.centre[0]+BUFFER,play_button.centre[1]), 5, (0,128,255), -1)
        ##################################################################
            
    else:
        if (time.time()-start_time < game_length):
            ### image processing code to obtaixn position of cat/player ###
            frame = get_processed_frame()										
            frame = crop_frame_to_screen(frame)
            found_object = findObject(frame, background)
            if found_object is not None:
                thresholded, largest_obj = found_object
                getOuterPoints(largest_obj)          
            
                ##################################################################
                if troubleshoot:
                    cv2.drawContours(marker_frame, [largest_obj], -1, (255,0,0), 3)
                ##################################################################
                    
            if cv2.waitKey(1) == 27:
                break
            ### end of image processing code ###
            
            gamescreen.blit(bg_square, (1200, 30))                  #cover score from previous frame
            enemy.update(tx, ty, lx, ly, rx, ry)
            scoreText = myfont.render("Score = "+str(score), 3, (255,0,0))
            gamescreen.blit(scoreText, (1120, 30))
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == ord('x'):
                        run = False
                    elif event.key == ord('r'):
                        start_time = time.time()
                        enemy.reset()
                                            
            pygame.display.update()
            
            #####################################################################
            if troubleshoot:
                cv2.circle(marker_frame, (tx,ty), 3, (255,0,0), -1)                 #uncomment to mark top-most point of object
                cv2.circle(marker_frame, (tx,ty), COL_RADIUS, (255,0,0), 3)         #uncomment to mark radius of collision ^
                cv2.circle(marker_frame, (lx,ly), 3, (0,255,0), -1)                 #uncomment to mark left-most point of object
                cv2.circle(marker_frame, (lx,ly), COL_RADIUS, (0,255,0), 3)         #uncomment to mark radius of collision ^
                cv2.circle(marker_frame, (rx,ry), 3, (0,0,255), -1)                 #uncomment to mark right-most point of object
                cv2.circle(marker_frame, (rx,ry), COL_RADIUS, (0,0,255), 3)
                cv2.circle(marker_frame, (mouse_x,mouse_y), 5, (0,128,255), -1)
            #####################################################################
                
        else:
            gamescreen.blit(bg_img, (0,0))
            restart_button.draw()
            exit_button.draw()
            if score > 1:
                gameOverText2 = myfont.render("VICTORY! THIS FEROCIOUS HUNTER SMACKED THE MOUSE "+str(score)+" TIMES",3,(0,0,0))
                gamescreen.blit(gameOverText2, (100, 360))
            elif score == 1:
                gameOverText1 = myfont.render("GOT 'EM! YOU CAUGHT THE MOUSE ONE TIME",3,(0,0,0))
                gamescreen.blit(gameOverText1, (300, 360))
            else:
                gameOverText0 = myfont.render("YOU DIDNT MANAGE TO CATCH THE MOUSE, YOU'LL GET 'EM NEXT TIME TIGER!",3,(0,0,0))
                gamescreen.blit(gameOverText0, (40, 360))
            pygame.display.update()
            frame = get_processed_frame()
            frame = crop_frame_to_btns(frame)
            found_object = findObject(frame,threshold=50, minSize = 200, background=button_background,blur=False)
            if found_object is not None:
                thresholded, largest_obj = found_object
                getOuterPoints(largest_obj)
                
                if restart_button.is_pressed():
                    start_time = time.time()
                    enemy.reset()
                elif exit_button.is_pressed():
                    run = False
                    
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == ord('x'):
                        run = False
                    elif event.key == ord('r'):
                        start_time = time.time()
                        enemy.reset()
                        
    ###########################################################################            
            if troubleshoot:
                cv2.circle(marker_frame, (tx+BUFFER,ty), 3, (255,0,0), -1)                 #uncomment to mark top-most point of object
                cv2.circle(marker_frame, (tx+BUFFER,ty), BUTTON_COL_RADIUS, (255,0,0), 3)         #uncomment to mark radius of collision ^
                cv2.circle(marker_frame, (bx+BUFFER,by), 3, (0,255,255), -1)                 #uncomment to mark top-most point of object
                cv2.circle(marker_frame, (bx+BUFFER,by), BUTTON_COL_RADIUS, (0,255,255), 3)
                cv2.circle(marker_frame, (lx+BUFFER,ly), 3, (0,255,0), -1)                 #uncomment to mark left-most point of object
                cv2.circle(marker_frame, (lx+BUFFER,ly), BUTTON_COL_RADIUS, (0,255,0), 3)         #uncomment to mark radius of collision ^
                cv2.circle(marker_frame, (rx+BUFFER,ry), 3, (0,0,255), -1)                 #uncomment to mark right-most point of object
                cv2.circle(marker_frame, (rx+BUFFER,ry), BUTTON_COL_RADIUS, (0,0,255), 3)
                cv2.circle(marker_frame, (restart_button.centre[0]+BUFFER,restart_button.centre[1]), 5, (0,128,255), -1)
                cv2.circle(marker_frame, (exit_button.centre[0]+BUFFER,exit_button.centre[1]), 5, (0,128,255), -1)
    if troubleshoot:
        cv2.imshow('Show markers', marker_frame)
        cv2.waitKey(1)
    #########################################################################
    
    clock.tick(25)
    print("fps=",clock.get_fps())
    
cv2.destroyAllWindows()
cap.release()
pygame.quit()