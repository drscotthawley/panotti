#! /usr/bin/env python
'''
HeadGames -  GUI for demo-ing binaural source localization
Author: Scott Hawley

"But if you want to win, you gotta learn how to play"--Foreigner, "Head Games"

Still under construction
Requirements:
    pygame
    librosa
'''
from __future__ import print_function
import numpy as np
import pygame
import pygame.gfxdraw
import math
import librosa
import os
import sys
sys.path.insert(0, '..')
from keras.models import load_model
from panotti.datautils import *
#from predict_class import predict_one
import glob
import random
import re
import colorsys
import argparse


BLACK = (0, 0, 0)
DARKGREY = (55,55,55)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)


def predict_one(signal, sr, class_names, model, weights_file="weights.hdf5"):
    X = make_layered_melgram(signal,sr)
    return model.predict_proba(X,batch_size=1,verbose=0)[0]


def get_wav_file_list(path="binaural/Samples/",shuffle=True):
    path = path.rstrip('/')   # remove any trailing slash
    file_list = glob.glob(path+"*/*.wav")
    if shuffle:
        random.shuffle(file_list)
    return file_list

def parse_class_string(filename):
    str = re.search('class.*\/',filename).group(0)[0:-1]
    str2 = re.search('-a.*\.',str).group(0)[2:-1]
    return float(str2)

def draw_head(screen,origin,screensize):
    color=DARKGREY
    cx, cy, scale  = origin[0],origin[1], int(screensize[1]/4)
    headsize = (int(scale*2/3),scale)
    rx, ry = int(headsize[0]/2), int(headsize[1]/2)

    #ears
    color=BLACK
    earw, earh = scale/10, scale/4
    earrx, earry = int(scale/20), int(scale/8)
    pygame.gfxdraw.filled_ellipse(screen, cx-rx, cy, earrx, earry, color)
    pygame.gfxdraw.aaellipse(screen, cx-rx, cy, earrx, earry, color)
    pygame.gfxdraw.filled_ellipse(screen, cx+rx, cy, earrx, earry, color)
    pygame.gfxdraw.aaellipse(screen, cx+rx, cy, earrx, earry, color)


    # head proper
    color = DARKGREY

    headbox = (cx-headsize[0]/2, int(cy-scale/2), headsize[0], headsize[1])
    pygame.gfxdraw.filled_ellipse(screen, cx, cy, rx, ry, color)
    pygame.gfxdraw.aaellipse(screen, cx, cy, rx, ry, color)
    #head = pygame.draw.ellipse(screen, color, headbox)
    #nose
    noserad = int(scale/15)
    nose = pygame.draw.circle(screen, color, (cx,int(headbox[1]+noserad/4)), noserad  )
    return


def draw_bounds(screen,origin,screensize,angles):
    n_az = len(angles)
    radius = int(screensize[1]*.5)-10
    width = int(2)
    color = BLUE
    x, y, r = origin[0], origin[1], radius
    boundary = pygame.gfxdraw.aacircle(screen, x, y, r, color)
    #boundary = pygame.draw.circle(screen, color, origin, radius, width)

    # draw a bunch of lines
    radian_sweep = 2*math.pi / n_az
    radian_start = -0.5*radian_sweep
    for i in range(n_az):               # draw a bunch of bounds
        rad = radian_start + i*radian_sweep
        startpos = origin
        endpos = (int(origin[0]-radius*math.sin(rad)), int(origin[1]+radius*math.cos(rad)))
        pygame.draw.line(screen, color, startpos, endpos)
    return

def draw_pie(screen,origin,screensize,angles,guess_az,r_fac=1.0,color=GREEN):
    n_az = len(angles)

    if guess_az is None:
        return
    cx, cy, r  = origin[0],origin[1], screensize[1]*.5-10
    b = r/5
    r = b + r_fac*(r-b)

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



def assign_sat(color, sout):
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    r,g,b = color[0]/255.0, color[1]/255.0, color[2]/255.0
    #print("r,g,b =",r,g,b)
    h, s, v = rgb_to_hsv(r,g,b)
    #print("h,s,v=",h,s,v," sout=",sout)
    s = sout
    r, g, b = hsv_to_rgb(h, s, v)
    r, g, b = int(255*r), int(255*g), int(255*b)
    return (r,g,b)


def draw_probs_text(screen,origin,screensize,angles,probs):
    radius = int(screensize[1]*.38)
    fontsize = int(radius/7)
    myfont = pygame.font.SysFont('arial', fontsize)
    for i in range(len(angles)):               # draw a bunch of bounds
        rad = angles[i] * math.pi/180
        if (probs[i]==np.max(probs)):
            color=MAGENTA
        else:
            color=BLUE
        text = myfont.render('{0:.3f}'.format(probs[i]).lstrip('0'), True, color)
        tw, th = text.get_width(), text.get_height()
        x = int( origin[0] + radius * math.sin(rad) - tw/2)
        y = int( origin[1] - radius * math.cos(rad) - th/2)
        screen.blit(text,(x,y))
    return

def draw_probs_pies(screen,origin,screensize,angles,probs,true_az):
    radius = int(screensize[1]*.38)
    guess_az = angles[ np.argmax(probs)]

    for i in range(len(angles)):               # draw a bunch of bounds
        rad = angles[i] * math.pi/180
        piecolor = assign_sat(GREEN,probs[i])
        draw_pie(screen,origin,screensize,angles,angles[i],color=piecolor )

    draw_pie(screen,origin,screensize,angles,true_az,r_fac=1.04,color=RED )
    draw_pie(screen,origin,screensize,angles,guess_az,color=assign_sat(GREEN,np.max(probs)))
    return



################ MAIN CODE #############
def do_pygame(n_az=12, weights_file="binaural/weights.hdf5", wavpath="binaural/Samples/"):
    # Define some colors

    # make a list of valid angles
    angles = []
    deg_sweep = 360/n_az
    for n in range(n_az):
        angles.append(-180+ n*deg_sweep)
    print("angles = ",angles)
    file_list=get_wav_file_list(path=wavpath)
    print("len of file_list in path ",wavpath," = ",len(file_list))
    class_names=angles  # get_class_names(path="binaural/Samples/", sort=True)

    # Load the model
    print("Loading model...")
    model = load_model(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)
    model.summary()


    pygame.init()
    pygame.font.init()

    # Set the width and height of the screen [width, height]
    screensize = (500, 500)
    screen = pygame.display.set_mode(screensize,pygame.RESIZABLE)

    pygame.display.set_caption("Head Games")

    # Loop until the user clicks the close button.
    print("Every time you click the image, it will load a new file")
    done = False

    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    guess_az = 0.0
    true_az = 0.0
    deg_inc = 360/n_az
    probs = None
    # -------- Main Program Loop -----------
    while not done:
        # --- Main event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.VIDEORESIZE:
                screensize = (event.w, event.h)
                screen = pygame.display.set_mode(screensize,pygame.RESIZABLE)

        # --- Game logic should go here
            if event.type == pygame.MOUSEBUTTONUP:
                guess_az = guess_az + deg_inc

                # load new file
                filenum = random.randint(0,len(file_list)-1)
                print("filenum = ",filenum)
                infile = file_list[filenum]
                print("infile = ",infile)
                signal, sr = librosa.load(infile, mono=False, sr=44100)   # librosa naturally makes mono from stereo btw
                probs  = predict_one(signal, sr, class_names, model, weights_file=weights_file)
                print("     probs = ",probs)
                guess_az = angles[ np.argmax(probs)]
                # get true az from filename
                true_az = parse_class_string(infile)
                print("     guess_az, true_az = ",guess_az,", ",true_az,sep="")


        # --- Screen-clearing code goes here

        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.

        # If you want a background image, replace this clear with blit'ing the
        # background image.
        screen.fill(WHITE)

        # --- Drawing code should go here
        origin = (int(screensize[0]/2),int(screensize[1]/2))
        #draw (text) probabilities for different angles
        if probs is not None:
            draw_probs_pies(screen,origin,screensize,angles,probs,true_az)
            draw_probs_text(screen,origin,screensize,angles,probs)

        draw_bounds(screen,origin,screensize,angles)
        draw_head(screen,origin,screensize)

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(3)

    # Close the window and quit.
    pygame.quit()


if __name__ == "__main__":
    print("headgames.py - Still under construction\n")
    if (not os.path.isdir("binaural")) or (not os.path.isdir("binaural/Samples")):
        print("\nYou need to run ./binaural_setup.sh first.")
        sys.exit(1)
    parser = argparse.ArgumentParser(description="evaluates network on testing dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="binaural/weights.hdf5")
    parser.add_argument('-n', '--naz', #type=argparse.string,
        help='Number of azimuthal bins', default=12)
    parser.add_argument('-d', '--dir', #type=argparse.string,
            help='Directory containing test .wav files', default='binaural/Samples')

    args = parser.parse_args()
    weights_file=args.weights
    n_az = int(args.naz)
    print("n_az = ",n_az)
    wavpath = args.dir
    if (not os.path.isfile(weights_file)):
        print("Error, can't find weights file "+weights_file)
        sys.exit(1)
    #TODO: First check for existence of files we need: binaural/, binaural/Samples, binaural/weights.hdf5
    do_pygame(n_az=n_az, weights_file=weights_file, wavpath=wavpath)
