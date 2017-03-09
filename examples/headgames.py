#! /usr/bin/env python
'''
HeadGames -  GUI for demo-ing binaural source localization
Author: Scott Hawley

Still under construction
'''

import pygame
import math
#import librosa

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

def draw_head(screen,origin,screensize):
    color=BLACK
    cx, cy, scale  = origin[0],origin[1], int(screensize[1]/4)
    headsize = (int(scale*2/3),scale)
    headbox = (cx-headsize[0]/2, int(cy-scale/2), headsize[0], headsize[1])
    head = pygame.draw.ellipse(screen, color, headbox)
    noserad = int(scale/15)
    nose = pygame.draw.circle(screen, color, (cx,int(headbox[1]+noserad/4)), noserad  )

    earw = scale/10
    earh = scale/4
    hw = headsize[0]
    earbox_l = (cx-hw/2-earw/2, cy-earh/2, earw, earh)
    earbox_r = (cx+hw/2-earw/2, earbox_l[1], earbox_l[2], earbox_l[3])

    ear_l = pygame.draw.ellipse(screen, color, earbox_l )
    ear_r = pygame.draw.ellipse(screen, color, earbox_r )
    return head
 

def draw_bounds(screen,origin,screensize,n_az):
    radius = int(screensize[1]*.5)-2
    width = int(2)
    color = RED
    boundary = pygame.draw.circle(screen, color, origin, radius, width)

    # draw a bunch of lines
    radian_sweep = 2*math.pi / n_az
    radian_start = -0.5*radian_sweep
    for i in range(n_az):               # draw a bunch of bounds
        rad = radian_start + i*radian_sweep
        startpos = origin
        endpos = (int(origin[0]-radius*math.sin(rad)), int(origin[1]+radius*math.cos(rad)))
        pygame.draw.line(screen, color, startpos, endpos)
    return

def draw_guess(screen,origin,screensize,n_az,guess_az):
    if guess_az is None:
        return
    color = GREEN
    cx, cy, r  = origin[0],origin[1], int(screensize[1]*.5)-2

    guess_az_rad = guess_az*math.pi/180
    # Start list of polygon points
    deg_sweep = 360.0 / n_az
    radian_sweep = 2*math.pi / n_az
    radian_start = guess_az_rad - 0.5*radian_sweep
    deg_inc = math.pi/180
    p = [(cx, cy)]
    for n in range(0,int(deg_sweep)+1):  # Get points on arc in 1-degree increments
        rad = radian_start + n*deg_inc
        x = cx + int(r*math.sin(rad))
        y = cy - int(r*math.cos(rad))
        p.append((x, y))
    p.append((cx, cy))
    # Draw pie segment
    if len(p) > 2:
        pygame.draw.polygon(screen, color, p)
    return


def do_pygame(n_az=12):
    # Define some colors
     
    pygame.init()
     
    # Set the width and height of the screen [width, height]
    screensize = (500, 500)
    screen = pygame.display.set_mode(screensize)
     
    pygame.display.set_caption("Head Games")
     
    # Loop until the user clicks the close button.
    done = False
     
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    guess_az = -90.0
    deg_inc = 360/n_az
    # -------- Main Program Loop -----------
    while not done:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
     
        # --- Game logic should go here
     
        # --- Screen-clearing code goes here
     
        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
     
        # If you want a background image, replace this clear with blit'ing the
        # background image.
        screen.fill(WHITE)
     
        # --- Drawing code should go here
        origin = (int(screensize[0]/2),int(screensize[1]/2))
        guess_az = guess_az + deg_inc
        draw_guess(screen,origin,screensize,n_az,guess_az)
        draw_bounds(screen,origin,screensize,n_az)

        draw_head(screen,origin,screensize)

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
     
        # --- Limit to 60 frames per second
        clock.tick(3)
     
    # Close the window and quit.
    pygame.quit()


if __name__ == "__main__":
    print("headgames.py - Still under construction, for now it just shows an animation")
    do_pygame(n_az=12)
