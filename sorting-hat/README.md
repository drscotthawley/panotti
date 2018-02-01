# Sorting H.A.T.

*A product of Team 15 at the HackMT Hackathon: Scott Hawley, Braden Carei, Daniel Ellis, Will Haase, Braiden King, Tyler Thomas. January 26-28, 2018:*  

*As with panotti itself, this system may not be ready for 'prime time' use by the general public.  Rather, GitHub hosting simply serves as a convenient place to share work.*

Sorting H.A.T. (Hosted Audio Tagger) is a cloud-based service that applies machine learning to the task of audio 'tagging'.
This task is computation-intensive and beyond the capabilities of typical laptops, which is why we use GPU (graphics processing units) hosted in the cloud!

Panotti Server can enable composers and producers to re-label their audio sample libraries
according to genre, instrument, or even personal preferences.

## Requirements:
- panotti (requires keras, tensorflow,...)
- flask

## To Run:

    $ python sorting-hat.py

This starts an HTTP host on port 8000.  Once it's running, point your web browser to the URL of the machine running sorting-hat.py,
(e.g. http://127.0.0.1:8000 for localhost).

## Screenshot:
![screenshot](screenshot.png)
