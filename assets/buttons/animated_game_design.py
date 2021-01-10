# IMPORTS SECTION
import pygame
from pygame.locals import *
import numpy as np
from math import sqrt
import random
import cv2
import time

# GLOBAL GAME VARIABLES
start_time = time.time()
cx = 0
cy = 0
score = 0
col_radius = 80
time_prd = 5
diagonal_vel = 2
axis_vel = int(sqrt(2*diagonal_vel*diagonal_vel))
screen_width = 1280
screen_height = 720
sprite_width = 100
sprite_height = 100
clock = pygame.time.Clock()

all_directions=[-4,-3,-2,-1,1,2,3,4]      #array of directions for enemy to travel
pos_x = [1,2,-4]                          #positive x directions
pos_y = [-2,-3,-4]                        #negative x directions
neg_x = [-1,-2,4]                         #positive y directions
neg_y = [2,3,4]                           #negative y directions

animation_cds = [8,6,4]

# FUNCTION DEFINITIONS SECTION

# function to calculate distance between points in 2D and 3D space
def distance(p,q):
    return sqrt(sum([(a-b)**2 for a,b in zip(p, q)]))

# initialize pygame
pygame.init()

#load images
bg_img = pygame.image.load('white.png')
restart_button = pygame.image.load('restart_button.png')
restart_button = pygame.transform.scale(restart_button, (50,50))

#load fonts
myfont = pygame.font.SysFont('Arial',30)

#load sounds
soundObj = pygame.mixer.Sound('squeak1.wav')

class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.x = x                                                             #get sprite x and y co-ordinates
        self.rect.y = y

    def draw(self):
        pos = pygame.mouse.get_pos()
        gamescreen.blit(self.image, self.rect)
        
class Enemy():
    def __init__(self, x, y):
        #lists for sprite images:
        self.images_right = []                                                      
        self.images_left = []
        self.images_up = []
        self.images_down = []
        self.index = 0                                                              #index for animation state
        for num in range (0,3):                                                     #load sprite images
            img_right = pygame.image.load(f'rat_right_{num}.png')
            img_right = pygame.transform.scale(img_right, (sprite_width,sprite_height))
            self.images_right.append(img_right)
            img_left = pygame.image.load(f'rat_left_{num}.png')
            img_left = pygame.transform.scale(img_left, (sprite_width,sprite_height))
            self.images_left.append(img_left)
            img_up = pygame.image.load(f'rat_up_{num}.png')
            img_up = pygame.transform.scale(img_up, (sprite_width,sprite_height))
            self.images_up.append(img_up)
            img_down = pygame.image.load(f'rat_down_{num}.png')
            img_down = pygame.transform.scale(img_down, (sprite_width,sprite_height))
            self.images_down.append(img_down)
        self.image = self.images_right[self.index]                                  #assign initial image
        self.rect = self.image.get_rect()                                           #get co-ordinates of rectangle containing sprite
        self.rect.x = x                                                             #get sprite x and y co-ordinates
        self.rect.y = y
        self.centre = (self.rect.x+sprite_width//2, self.rect.y+sprite_height//2)   #get centre of sprite
        self.direction = 0                                                          #initialise direction of movement
        self.vel = 1                                                                #initialise velocity scaler
        self.score_cd = 0                                                           #cooldown before score can be updated (initially zero                                                          
        self.direction_cd = 5                                                       #cooldown before direction can change
        self.animation_cd = 8                                                       #initial cooldown for animation transitions
        self.cur_animation_cd = 8                                                   #cooldown for animation transitions updated based on current velocity
        
    def update(self, cx, cy):
        global score
        dx = 0
        dy = 0
        
        self.direction_cd -= 1                                                      #decrement cd on direction change
        if self.score_cd != 0:                                                      #decrement score cd unless it is already 0
            self.score_cd -= 1
        if self.direction_cd == 0:                                                  #if direction cd is 0
            self.direction = random.randint(-4,4)                                   #get new randomized direction
            self.vel = random.randint(2,4)                                          #get new randomized velocity
            self.cur_animation_cd = animation_cds[self.vel-2]                       #update animation transition rate based on new velocity
            interval = random.randint(1,3)                                          #get randomised time period for traveling in this new direction
            self.direction_cd = interval*time_prd
            
        dist = distance(self.centre,(cx,cy))                                        #check distance between object and cat
        if dist < col_radius:                                                       #if distance is smaller than the defined collision radius:
            if self.score_cd == 0:                                                  #update the score, play sound
                soundObj.play(0)
                self.score_cd = 30
                score += 1
                cur_direction = self.direction
                while self.direction == cur_direction:                              #change direction
                    self.direction = all_directions[random.randint(0,7)]
            self.vel = 5                                                            #set velocity to maximum
            self.direction_cd = 5*time_prd                                          #set time period of travel to custom maximum

        
        #define directional movement and animations
        if self.direction == 1:         #right
            dx = axis_vel
            dy = 0
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                self.image = self.images_right[self.index]
                
        elif self.direction == 2:       #up+right
            dx = diagonal_vel
            dy = -diagonal_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                self.image = self.images_right[self.index]
                
        elif self.direction == 3:       #up
            dx = 0
            dy = -axis_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_up):
                    self.index = 0
                self.image = self.images_up[self.index]
                
        elif self.direction == 4:       #up+left
            dx = -diagonal_vel
            dy = -diagonal_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -1:      #left
            dx = -axis_vel
            dy = 0
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -2:      #down+left
            dx = -diagonal_vel
            dy = diagonal_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_left):
                    self.index = 0
                self.image = self.images_left[self.index]
                
        elif self.direction == -3:      #down
            dx = 0
            dy = axis_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_down):
                    self.index = 0
                self.image = self.images_down[self.index]
                
        elif self.direction == -4:      #down+right
            dx = diagonal_vel
            dy = diagonal_vel
            self.animation_cd -= 1
            if self.animation_cd == 0:
                self.animation_cd = self.cur_animation_cd
                self.index += 1
                if self.index >= len(self.images_right):
                    self.index = 0
                self.image = self.images_right[self.index]

        #check for collisions and update position
        if (self.rect.x + dx*self.vel) < 0:                 #check for collision with left of screen
            self.direction = pos_x[random.randint(0,2)]     
            self.direction_cd = 5*time_prd
        elif (self.rect.x + dx*self.vel) > screen_width:    #check for collision with right of screen
            self.direction = neg_x[random.randint(0,2)]
            self.direction_cd = 5*time_prd
        else:
            self.rect.x += dx*self.vel                      
        if (self.rect.y + dy*self.vel) < 0:                 #check for collision with top of screen
            self.direction = pos_y[random.randint(0,2)]
            self.direction_cd = 5*time_prd
        elif (self.rect.y + dy*self.vel) > screen_height:   #check for collision with bottom of screen
            self.direction = neg_y[random.randint(0,2)]
            self.direction_cd = 5*time_prd
        else:
            self.rect.y += dy*self.vel

        #calculate new centre    
        self.centre = (self.rect.x+sprite_width//2, self.rect.y+sprite_height//2)
        gamescreen.blit(self.image, self.rect)
        
enemy = Enemy(100,100)
restart_button = Button(50,50,restart_button)

gamescreen = pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption('Mouse Hunter')

# MAIN LOOP SECTION
time.clock()
run = True
while run:
    if (time.time() - start_time < 10):
        gamescreen.blit(bg_img, (0,0))
        restart_button.draw()
        cx,cy = pygame.mouse.get_pos()
        print((cx,cy))
        print(enemy.centre)
        enemy.update(cx, cy)
        scoreText = myfont.render("Score = "+str(score), 3, (255,0,0))
        gamescreen.blit(scoreText, (1120, 30))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == ord('q'):
                    run = False
        clock.tick(30)           
        pygame.display.update()
        print("fps=",clock.get_fps())
    else:
        run = False
    
pygame.quit()
