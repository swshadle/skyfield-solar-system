#!/usr/bin/env python
# coding: utf-8

# This is a [Bokeh app](https://docs.bokeh.org/en/latest/) showing line-of-sight from Earth to other planets, illustrating the cause of retrograde motion and annotating the dates of retrograde motion and opposition. Retrograde motion is illustrated with a red line.
# 
# I wrote this with [Anaconda](https://www.anaconda.com) on my Mac. This script uses an ephemeris table from [Skyfield](https://rhodesmill.org/skyfield/planets.html) for solar system positions.
# 
# Although I probably can't help you debug your program I would love to hear if you use this animation (or derivative) in your astronomy lessons or observations.
# 
# Good luck,
# 
# Stephen Shadle 🌌
# 
# swshadle@gmail.com

# # Versions
# 
# Skyfield-Bokeh_with_Table: Renamed because I deleted -Copy6 and File->Make a Copy keeps going back to it. Can row selection show the upcoming events (so clicking on a row doesn't make it disappear) but the animation and normal slider selection only show past events?
# 
# Skyfield_with_Events-Bokeh-Copy8: Remove debugging text from label and slider. General code cleanup.
# 
# Skyfield_with_Events-Bokeh-Copy7: Debug dates being off. Sometimes the datatable, slider value and label on the plot show different dates. Why??? Adding integer values to all 3 displays to compare what's behind the different dates. Slider.value's auto datetime seems always off. Hiding the display (show_value=False) and just using slider.title to display the date.
# 
# Skyfield_with_Events-Bokeh-Copy6: loading message--failed. Couldn't get Bokeh to display the plots while loading the database.
# 
# Skyfield_with_Events-Bokeh-Copy5: Added retrograde line color, background stars and datatable. Updated hovertool for planets and removed for background stars. Next: loading message.
# 
# Skyfield_with_Events-Bokeh-Copy4: Got line-of-sight lines from earth.
# 
# Skyfield_with_Events-Bokeh-Copy3: Got plot working with dates in the slider. Added a simple line to connect the planets (needs to be replaced with line-of-sight lines from earth)
# 
# Skyfield Sky View & Retrograde with Events-Bokeh: Getting ready for Bokeh plots, animations, and server
# 
# Skyfield Sky View & Retrograde with Events-Colab-2: Adding file saving back in now that the inline animation is working
# 
# Skyfield Sky View & Retrograde with Events-Colab-1: Got animation working, added Timer function and status bar
# 
# Skyfield Sky View & Retrograde with Events-Colab: Make a version that animates in Google Colab
# 
# Future: Only show mars's motion line during retrograde (and other planets').
# 
# Skyfield Sky View & Retrograde with Events 4: Remove Jupiter/Saturn begins/ends retrograde from events list. Just show opposition. Change aspect ration of left panel (ax1) to be taller. Include more stars (or entire field of view), not just limited to the planets' motion through the ecliptic.
# 
# Skyfield Sky View & Retrograde with Events 3: Back up date to include start of 2020 retrograde & oppositions for jupiter and saturn. Change saturn's marker symbol to a more ring-like circle with a slash through it ($\emptyset$). $\frac{h}{2\pi} = \hbar$
# 
# Skyfield Sky View & Retrograde with Events 2: Add line-of-sight line for Jupiter and Saturn
# 
# Skyfield Sky View & Retrograde with Events 1: Use add_gridspec to control plot layout within the figure
# 
# Skyfield Sky View & Retrograde with Events: Name change reflects merger of this script with Skyfield Retrograde Inner Planets with Events, the single-panel script that includes event notation text to the right of the animation.
# 
# Skyfield Retrograde Named Stars w Inner Planets: original version.

# %pwd


# to run a local bokeh server, at the command line/terminal:<br>
# * change to current working directory (from `%pwd`) with<br>
# `cd ~/Documents/_Python/Jupyter_Notebooks/Bokeh/`<br>
# 
# * run with<br>
# `bokeh serve --show Skyfield-Bokeh_with_Table.ipynb`

# to convert to .py for uploading to a hosting service, run at the command line/terminal:
# jupyter nbconvert --to script --no-prompt Skyfield-Bokeh_with_Table.ipynb


# Suppress warnings. Comment this out if you wish to see the warning messages
import warnings

warnings.filterwarnings('once') # warnings are on with 'default' and off with 'ignore'
# print('warnings suppressed')


# %pip install skyfield
# %pip install --upgrade skyfield


# use "%pip install" to load skyfield api once then comment this line out
###### %pip install skyfield
###### %pip install --upgrade skyfield
from skyfield.api import Star, load, Topos
from skyfield.data import hipparcos
import numpy as np
import pandas as pd
import ipywidgets as widgets
# print('modules loaded')


from skyfield import __version__ as skyfieldversion
print (f'tested with skyfield version 1.30. currently using: {skyfieldversion}')


from datetime import datetime

class Timer(object):
    def __init__(self, total):
        self.start = datetime.now()
        self.total = total
 
    def remains(self, done):
        now  = datetime.now()
        left = (self.total - done) * (now - self.start) / done
    
        da, remainder  = divmod(left.total_seconds(), 24*3600)
        hrs, remainder = divmod(remainder, 3600)
        mins, secs = divmod(remainder, 60)

        if da:
            return f'{int(da)} days {int(hrs)} hours {int(mins)} minutes {int(secs)} seconds remaining'     
        elif hrs:
            return f'{int(hrs)} hours {int(mins)} minutes {int(secs)} seconds remaining                '
        elif mins:
            return f'{int(mins)} minutes {int(secs)} seconds remaining                  '
        else:
            return f'{int(secs)} seconds remaining                     '

def timer_start():
    global start_time
    start_time = datetime.now()

def timer_stop(fn=None):
    time_elapsed = datetime.now() - start_time

    if fn:
        print(f'{fn}:')

    da, remainder  = divmod(time_elapsed.total_seconds(), 24*3600)
    hrs, remainder = divmod(remainder, 3600)
    mins, secs = divmod(remainder, 60)

    if da:
      print(f'{int(da)} days {int(hrs)} hours {int(mins)} minutes {int(secs)} seconds elapsed')
    elif hrs:
      print(f'{int(hrs)} hours {int(mins)} minutes {int(secs)} seconds elapsed')
    elif mins:
      print(f'{int(mins)} minutes {int(secs)} seconds elapsed')
    elif secs >= 1.0:
      print(f'{int(secs)} seconds elapsed')
    else:
      print(f'{secs:.2} seconds elapsed')
        
# print('timer functions loaded')


try:
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
except IOError as e:
    print(e)
    print('looking for a local copy stored in "data" subdirectory')
    try:
        with load.open('./data/hip_main.dat.gz') as f:
            df = hipparcos.load_dataframe(f)
    except IOError as e:
        print(e)
    except:
        print('unknown error opening local copy')
except:
    print('unknown error opening', hipparcos.URL)
finally:
    print('dataframe loaded successfully')


# If that worked with no errors, great! If not, the following cell has one last method.<br>
# (This only works in Google Colab!)

# # load hip_main.dat.gz from a local drive--this works in Google Colab
# from google.colab import files

# uploaded = files.upload()

# for fn in uploaded.keys():
#   print('User uploaded file "{name}" with length {length} bytes'.format(
#       name=fn, length=len(uploaded[fn])))

# if len(uploaded)==1:
#     with load.open(fn) as f:
#         df = hipparcos.load_dataframe(f)
#     print('dataframe loaded successfully')


# # load skyfield ephemeris table from local directory (see https://rhodesmill.org/skyfield/planets.html)
eph = load('de421.bsp') # ephemeris table de421.bsp is only valid through 2053 Oct 9 (Julian date 2471184.5)


sun     = eph['Sun']
mercury = eph['Mercury']
venus   = eph['Venus']
earth   = eph['Earth']
mars    = eph['Mars']
jupiter = eph['Jupiter barycenter']
saturn  = eph['Saturn barycenter']
# uranus  = eph['Uranus barycenter']
# neptune = eph['Neptune barycenter']
# pluto   = eph['Pluto barycenter']


ts = load.timescale()

# hours = (3*365+1)*24 # how long the animation should run (number of hours of data from ephemeris table)
hours = (4*365+2) # how long the animation should run (although `hours` is used throughout the script, each one represents one day or 24 hours of data from the ephemeris table))

start_time = datetime.now()
start_y = start_time.year  # 2020 (use numbers here to select a fixed start date rather than today's date)
start_m = start_time.month # 5
start_d = start_time.day   # 11

# t = ts.utc(start_y,start_m,start_d,range(hours))
t = ts.utc(start_y, start_m, # starting year and month
           range(start_d, start_d+hours), # using a range in the days position results in a list. here, the use of `hours` actually means the number of days that we want
           0, 0, 0.0) # starting hours, minutes and seconds


numremoved = 0

print(f'Solar system dates from {t[0].utc_datetime():%d %b %Y} to {t[-1].utc_datetime():%d %b %Y}')
while len(t)>1 and t[-1].tt>2471184.5: # make sure the dates don't go past the useful info from ephemeris table
    t=t[:-1]
    numremoved += 1

if numremoved:
    print(f'removed {numremoved} frames')
    hours -= numremoved
    t = ts.utc(start_y,start_m,range(start_d, start_d+hours),0,0,0.0)
    print(f'Corrected dates from {t[0].utc_datetime():%d %b %Y} to {t[-1].utc_datetime():%d %b %Y}')
    print(f'Ephemeris table only valid through {eph.comments()[359:370]}')

# assert an error if there is no remaining valid data
assert t[-1].tt<2471184.5, 'Dates are out of range'
assert hours>0, f'too few hours selected: {hours}'

sun_p     = sun.at(t).ecliptic_position().au
mercury_p = mercury.at(t).ecliptic_position().au
venus_p   = venus.at(t).ecliptic_position().au
earth_p   = earth.at(t).ecliptic_position().au
# moon_p    = moon.at(t).ecliptic_position().au
mars_p    = mars.at(t).ecliptic_position().au
jupiter_p = jupiter.at(t).ecliptic_position().au
saturn_p  = saturn.at(t).ecliptic_position().au
# uranus_p  = uranus.at(t).ecliptic_position().au
# neptune_p = neptune.at(t).ecliptic_position().au
# pluto_p   = pluto.at(t).ecliptic_position().au


sun_x,     sun_y,     sun_z     = sun_p
mercury_x, mercury_y, mercury_z = mercury_p
venus_x,   venus_y,   venus_z   = venus_p
earth_x,   earth_y,   earth_z   = earth_p
mars_x,    mars_y,    mars_z    = mars_p
jupiter_x, jupiter_y, jupiter_z = jupiter_p
saturn_x,  saturn_y,  saturn_z  = saturn_p


data_x = np.stack((sun_x, mercury_x, venus_x, earth_x, mars_x, jupiter_x, saturn_x))
data_y = np.stack((sun_y, mercury_y, venus_y, earth_y, mars_y, jupiter_y, saturn_y))
# data_z = np.stack((sun_z, mercury_z, venus_z, earth_z, mars_z, jupiter_z, saturn_z))


from math import atan2
def getAngleRad(x1, y1, x2, y2): # returns the angle between 2 points in radians
    return atan2(y2 - y1, x2 - x1)


data_mercury_rays = [getAngleRad(x1, y1, x2, y2) for (x1, y1, x2, y2) in zip(earth_x, earth_y, mercury_x, mercury_y)]
data_venus_rays   = [getAngleRad(x1, y1, x2, y2) for (x1, y1, x2, y2) in zip(earth_x, earth_y, venus_x, venus_y)]
data_mars_rays    = [getAngleRad(x1, y1, x2, y2) for (x1, y1, x2, y2) in zip(earth_x, earth_y, mars_x, mars_y)]
data_jupiter_rays = [getAngleRad(x1, y1, x2, y2) for (x1, y1, x2, y2) in zip(earth_x, earth_y, jupiter_x, jupiter_y)]
data_saturn_rays  = [getAngleRad(x1, y1, x2, y2) for (x1, y1, x2, y2) in zip(earth_x, earth_y, saturn_x, saturn_y)]


data_rays = np.stack((data_mercury_rays, data_venus_rays, data_mars_rays, data_jupiter_rays, data_saturn_rays))


index5 = ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']
df_rays = pd.DataFrame(data=data_rays, index=index5)


data_earth_x = np.stack((earth_x, earth_x, earth_x, earth_x, earth_x))
data_earth_y = np.stack((earth_y, earth_y, earth_y, earth_y, earth_y))


df_earth_x = pd.DataFrame(data=data_earth_x, index=index5)
df_earth_y = pd.DataFrame(data=data_earth_y, index=index5)


xmax = max(abs(data_x.min()), abs(data_x.max()))
ymax = max(abs(data_y.min()), abs(data_y.max()))


# get latitude, longitude, distance for planets
mercuryeclat, mercury_eclon, mercuryecd = earth.at(t).observe(mercury).ecliptic_latlon()
mercury_eclondel = mercury_eclon.radians[1:] - mercury_eclon.radians[:-1]

venus_eclat, venus_eclon, venus_ecd = earth.at(t).observe(venus).ecliptic_latlon()
venus_eclondel = venus_eclon.radians[1:] - venus_eclon.radians[:-1]

mars_eclat, mars_eclon, mars_ecd = earth.at(t).observe(mars).ecliptic_latlon()
mars_eclondel = mars_eclon.radians[1:] - mars_eclon.radians[:-1]

jupiter_eclat, jupiter_eclon, jupiter_ecd = earth.at(t).observe(jupiter).ecliptic_latlon()
jupiter_eclondel = jupiter_eclon.radians[1:] - jupiter_eclon.radians[:-1]

saturn_eclat, saturn_eclon, saturn_ecd = earth.at(t).observe(saturn).ecliptic_latlon()
saturn_eclondel = saturn_eclon.radians[1:] - saturn_eclon.radians[:-1]





# combine into arrays with 2 columns to keep dates and motion in synch during data cleaning. column 0 is delta ecliptic longitude and column 1 is dates
mercury_data2d = np.stack((mercury_eclondel, t[:-1]), axis=1)
venus_data2d   = np.stack((venus_eclondel,   t[:-1]), axis=1)
mars_data2d    = np.stack((mars_eclondel,    t[:-1]), axis=1)
jupiter_data2d = np.stack((jupiter_eclondel, t[:-1]), axis=1)
saturn_data2d  = np.stack((saturn_eclondel,  t[:-1]), axis=1)

# data cleaning
def reject_outliers_2d(data, m=2.0): # for a 2d array
    d = np.abs(data[:,0] - np.median(data[:,0]))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.0
    return data[s<m]

filtered_mercury_data2d = reject_outliers_2d(mercury_data2d,200.)
filtered_venus_data2d   = reject_outliers_2d(venus_data2d,200.)
filtered_mars_data2d    = reject_outliers_2d(mars_data2d,200.)
filtered_jupiter_data2d = reject_outliers_2d(jupiter_data2d,200.)
filtered_saturn_data2d  = reject_outliers_2d(saturn_data2d,200.)

mercury_prograde = filtered_mercury_data2d[:,0] >= 0.0
venus_prograde   = filtered_venus_data2d[:,0]   >= 0.0
mars_prograde    = filtered_mars_data2d[:,0]    >= 0.0
jupiter_prograde = filtered_jupiter_data2d[:,0] >= 0.0
saturn_prograde  = filtered_saturn_data2d[:,0]  >= 0.0


# since we are likely to have shortened our planetary position arrays during data cleaning, 
# now we'll tack the same value onto the end to get our original array length back

if not len(mercury_eclon.radians) == len(venus_eclon.radians) == len(mars_eclon.radians) == len(jupiter_eclon.radians) == len(saturn_eclon.radians):
    raise ValueError("oops, shouldn't have different lengths of planet data, e.g., mercury_eclon")

print('cleaning glitchy data')
while len(mercury_prograde) < len(mercury_eclon.radians):
    print('¯\_(ツ)_/¯ mercury')
    mercury_prograde = np.append(mercury_prograde,mercury_prograde[-1])

while len(venus_prograde) < len(venus_eclon.radians):
    print('¯\_(ツ)_/¯ venus')
    venus_prograde = np.append(venus_prograde,mars_prograde[-1])

while len(mars_prograde) < len(mars_eclon.radians):
    print('¯\_(ツ)_/¯ mars')
    mars_prograde = np.append(mars_prograde,mars_prograde[-1])

while len(jupiter_prograde) < len(jupiter_eclon.radians):
    print('¯\_(ツ)_/¯ jupiter')
    jupiter_prograde = np.append(jupiter_prograde,jupiter_prograde[-1])

while len(saturn_prograde) < len(saturn_eclon.radians):
    print('¯\_(ツ)_/¯ saturn')
    saturn_prograde = np.append(saturn_prograde,saturn_prograde[-1])


# # store an array of colors (one for prograde and a contrasting one for retrograde)
procolor = 'blue' # or maybe 'grey', 'darkblue'
retcolor = 'orangered'
mercury_linecolor = [procolor if mercury_prograde[i] else retcolor for i in range(len(mercury_prograde))]
venus_linecolor   = [procolor if venus_prograde[i]   else retcolor for i in range(len(venus_prograde))]
mars_linecolor    = [procolor if mars_prograde[i]    else retcolor for i in range(len(mars_prograde))]
jupiter_linecolor = [procolor if jupiter_prograde[i] else retcolor for i in range(len(jupiter_prograde))]
saturn_linecolor  = [procolor if saturn_prograde[i]  else retcolor for i in range(len(saturn_prograde))]


data_linecolor = np.stack((mercury_linecolor,
                           venus_linecolor,
                           mars_linecolor,
                           jupiter_linecolor,
                           saturn_linecolor))


df_linecolor = pd.DataFrame(data=data_linecolor, index=index5)


df2 = pd.concat({
    'earth_x': df_earth_x,
    'earth_y': df_earth_y,
    'angle': df_rays,
    'color': df_linecolor
    }, axis=1)


# make a structured array (events) to hold a list of notable events (i.e., begin/end retrograde & opposition) for the planets
mercury_motion_chg = np.where(mercury_prograde[:-1] != mercury_prograde[1:])[0]+1
venus_motion_chg   = np.where(venus_prograde[:-1]   != venus_prograde[1:])[0]+1
mars_motion_chg    = np.where(mars_prograde[:-1]    != mars_prograde[1:])[0]+1
jupiter_motion_chg = np.where(jupiter_prograde[:-1] != jupiter_prograde[1:])[0]+1
saturn_motion_chg  = np.where(saturn_prograde[:-1]  != saturn_prograde[1:])[0]+1


# make a structured array (events) to hold a list of notable events (i.e., begin/end retrograde & opposition) for the planets
dtype = [('Frame', np.int32), ('Epoch', np.int32), ('Date', (np.str_, 10)), ('Text', (np.str_, 25))]
events = np.array([], dtype=dtype)


# events


# construct list of events for each planet

# # mercury: (no such thing as opposition for inferior planets)
# for i in mercury_motion_chg:
#     if not mercury_prograde[i]: # not prograde means entering retrograde
#         e = int(filtered_mercury_data2d[i,1].utc_datetime().timestamp())
#         events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_mercury_data2d[i,1].utc_datetime()),"Mercury begins retrograde")], dtype=dtype)), axis=0)
# #         print('Mercury begins retrograde {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_mercury_data2d[i,1].utc_datetime(), i, e))
#     else: # entering prograde
#         e = int(filtered_mercury_data2d[i,1].utc_datetime().timestamp())
#         events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_mercury_data2d[i,1].utc_datetime()),"Mercury ends retrograde")], dtype=dtype)), axis=0)
# #         print('Mercury ends retrograde   {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_mercury_data2d[i,1].utc_datetime(), i, e))

# # venus: (no such thing as opposition for inferior planets)
# for i in venus_motion_chg:
#     if not venus_prograde[i]: # not prograde means entering retrograde
#         e = int(filtered_venus_data2d[i,1].utc_datetime().timestamp())
#         events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_venus_data2d[i,1].utc_datetime()),"Venus begins retrograde")], dtype=dtype)), axis=0)
# #         print('Venus begins retrograde {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_venus_data2d[i,1].utc_datetime(), i, e))
#     else: # entering prograde
#         e = int(filtered_venus_data2d[i,1].utc_datetime().timestamp())
#         events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_venus_data2d[i,1].utc_datetime()),"Venus ends retrograde")], dtype=dtype)), axis=0)
# #         print('Venus ends retrograde   {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_venus_data2d[i,1].utc_datetime(), i, e))

# mars:
opposition = 0 # the date of opposition is calculated as occuring midway through beginning and end of retrograde motion
opposition_epoch = 0 # calculate halfway mark of retrograde motion using epoch seconds
for i in mars_motion_chg:
    if not mars_prograde[i]: # not prograde means entering retrograde
        opposition = i
        opposition_epoch = int(filtered_mars_data2d[i,1].utc_datetime().timestamp()) # start counting epoch seconds from beginning of retrograde motion
        e = int(filtered_mars_data2d[i,1].utc_datetime().timestamp())
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_mars_data2d[i,1].utc_datetime()),"Mars begins retrograde")], dtype=dtype)), axis=0)
#         print('Mars begins retrograde {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_mars_data2d[i,1].utc_datetime(), i, e))
    else: # entering prograde
        if opposition != 0: # don't bother calculating opposition if we didn't enter retrograde (at the beginning of the simulation we could already be in retrograde)
            opposition = (opposition + i) // 2 # integer floor division
            opposition_epoch = opposition_epoch + (int(filtered_mars_data2d[i,1].utc_datetime().timestamp()) - opposition_epoch) // 2 
            events = np.concatenate((events, np.array([(opposition,opposition_epoch,'{:%Y-%m-%d}'.format(filtered_mars_data2d[opposition,1].utc_datetime()),"Mars at opposition")], dtype=dtype)), axis=0)
#             print('Mars at opposition     {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_mars_data2d[opposition,1].utc_datetime(), opposition, opposition_epoch))
        e = int(filtered_mars_data2d[i,1].utc_datetime().timestamp())
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_mars_data2d[i,1].utc_datetime()),"Mars ends retrograde")], dtype=dtype)), axis=0)
#         print('Mars ends retrograde   {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_mars_data2d[i,1].utc_datetime(), i, e))

'''
Note that jupiter and saturn beginning and ending retrograde makes for a long list
of events since these occur approximately every year (compared with every 2 years
for mars). If you wish to include these events, uncomment them in the following code.
'''
# jupiter:
opposition = 0 # the date of opposition is calculated as occuring midway through beginning and end of retrograde motion
opposition_epoch = 0 # calculate halfway mark of retrograde motion using epoch seconds
for i in jupiter_motion_chg:
    if not jupiter_prograde[i]: # not prograde means entering retrograde
        opposition = i
        opposition_epoch = int(filtered_jupiter_data2d[i,1].utc_datetime().timestamp()) # start counting epoch seconds from beginning of retrograde motion
        e = int(filtered_jupiter_data2d[i,1].utc_datetime().timestamp())
        # comment the following line to keep retrograde off the list of events
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_jupiter_data2d[i,1].utc_datetime()),"Jupiter begins retrograde")], dtype=dtype)), axis=0)
#         print('Jupiter begins retrograde {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_jupiter_data2d[i,1].utc_datetime(), i, e))
    else: # entering prograde
        if opposition != 0: # don't bother calculating opposition if we didn't enter retrograde (at the beginning of the simulation we could already be in retrograde)
            opposition = (opposition + i) // 2 # integer floor division
            opposition_epoch = opposition_epoch + (int(filtered_jupiter_data2d[i,1].utc_datetime().timestamp()) - opposition_epoch) // 2 
            events = np.concatenate((events, np.array([(opposition,opposition_epoch,'{:%Y-%m-%d}'.format(filtered_jupiter_data2d[opposition,1].utc_datetime()),"Jupiter at opposition")], dtype=dtype)), axis=0)
#             print('Jupiter at opposition     {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_jupiter_data2d[opposition,1].utc_datetime(), opposition, opposition_epoch))
        e = int(filtered_jupiter_data2d[i,1].utc_datetime().timestamp())
        # comment the following line to keep retrograde off the list of events
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_jupiter_data2d[i,1].utc_datetime()),"Jupiter ends retrograde")], dtype=dtype)), axis=0)
#         print('Jupiter ends retrograde   {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_jupiter_data2d[i,1].utc_datetime(), i, e))

# saturn:
opposition = 0 # the date of opposition is calculated as occuring midway through beginning and end of retrograde motion
opposition_epoch = 0 # calculate halfway mark of retrograde motion using epoch seconds
for i in saturn_motion_chg:
    if not saturn_prograde[i]: # not prograde means entering retrograde
        opposition = i
        opposition_epoch = int(filtered_saturn_data2d[i,1].utc_datetime().timestamp()) # start counting epoch seconds from beginning of retrograde motion
        e = int(filtered_saturn_data2d[i,1].utc_datetime().timestamp())
        # comment the following line to keep retrograde off the list of events
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_saturn_data2d[i,1].utc_datetime()),"Saturn begins retrograde")], dtype=dtype)), axis=0)
#         print('Saturn begins retrograde {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_saturn_data2d[i,1].utc_datetime(), i, e))
    else: # entering prograde
        if opposition != 0: # don't bother calculating opposition if we didn't enter retrograde (at the beginning of the simulation we could already be in retrograde)
            opposition = (opposition + i) // 2 # integer floor division
            opposition_epoch = opposition_epoch + (int(filtered_saturn_data2d[i,1].utc_datetime().timestamp()) - opposition_epoch) // 2 
            events = np.concatenate((events, np.array([(opposition,opposition_epoch,'{:%Y-%m-%d}'.format(filtered_saturn_data2d[opposition,1].utc_datetime()),"Saturn at opposition")], dtype=dtype)), axis=0)
#             print('Saturn at opposition     {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_saturn_data2d[opposition,1].utc_datetime(), opposition, opposition_epoch))
        e = int(filtered_saturn_data2d[i,1].utc_datetime().timestamp())
        # comment the following line to keep retrograde off the list of events
        events = np.concatenate((events, np.array([(i,e,'{:%Y-%m-%d}'.format(filtered_saturn_data2d[i,1].utc_datetime()),"Saturn ends retrograde")], dtype=dtype)), axis=0)
#         print('Saturn ends retrograde   {:%Y-%m-%d} frame #{:04} epoch {}'.format(filtered_saturn_data2d[i,1].utc_datetime(), i, e))


# sort events by Frame number
events = np.sort(events, order='Epoch')


index7 = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn']
df_x = pd.DataFrame(data=data_x, index=index7)
df_y = pd.DataFrame(data=data_y, index=index7)
# df_z = pd.DataFrame(data=data_z, index=index7)


data_color = ['yellow', 'burlywood', 'blanchedalmond', 'mediumturquoise', 'lightcoral', 'burlywood', 'seashell']
data_size = [7, 2, 4, 4, 3, 6, 6]
data_size = [x * 2 for x in data_size]
df_color = pd.DataFrame(data=data_color, index=index7, columns=[''])#, columns=['color'])
df_size = pd.DataFrame(data=data_size, index=index7, columns=[''])#, columns=['size'])


df = pd.concat({
    'x': df_x,
    'y': df_y,
#     'z': df_z,
    }, axis=1)


# display progress bar in the console
import sys
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = count / float(total)
    bar = '█' * filled_len + '·' * (bar_len - filled_len)
    sys.stdout.write(f'{bar} {percents:.1%} done. {status}\r')
    sys.stdout.flush()

# There are `hours` units in this task
timer = Timer(hours)
data  = {}
data2 = {}
for hour in range(hours):
    progress(hour, hours, status=f'{timer.remains(hour+1)}')
    df_hour = df.iloc[:,df.columns.get_level_values(1)==hour]
    df_hour.columns = df_hour.columns.droplevel(1)
    data[hour] = df_hour.join(df_size).reset_index()
#     data[hour].rename(columns={'index': 'Body'}, inplace=True)
#     data[hour].rename(columns={'': 'size'}, inplace=True)
    data[hour].columns = ['Body', 'x', 'y', 'size']
    df2_hour = df2.iloc[:,df2.columns.get_level_values(1)==hour]
    df2_hour.columns = df2_hour.columns.droplevel(1)
    data2[hour] = df2_hour.reset_index()
sys.stdout.write('\n')


# from bokeh.layouts import layout
from bokeh.layouts import column, row
from bokeh.models import (Button, CategoricalColorMapper, ColumnDataSource, Div,
                          HoverTool, Label, SingleIntervalTicker, DateSlider,
                          Legend, LegendItem, Range1d, DataTable, TableColumn)
from bokeh.palettes import Spectral7
# from bokeh.palettes import Turbo256
from bokeh.plotting import figure, curdoc, show
# from bokeh.themes import built_in_themes
from bokeh.io import output_notebook
print('tested with BokehJS 2.2.0')
output_notebook()


# done with database prep
eph.close()
source  = ColumnDataSource(data[0])
source2 = ColumnDataSource(data2[0])


# source_tab = ColumnDataSource(pd.DataFrame(events[events['Frame']==0]))
source_tab = ColumnDataSource(pd.DataFrame(events[:1]))
table_cols = [
        TableColumn(field='Frame',title='Frame', width=50),
#         TableColumn(field='Epoch',title='Frame', width=80),
        TableColumn(field='Date', title='Date',  width=70),# formatter=DateFormatter()),
        TableColumn(field='Text', title='Event', width=160),
]
    
data_table = DataTable(source=source_tab, columns=table_cols, index_position=None, row_height=20,
                       reorderable=False, sortable=False, selectable=True, fit_columns=False,
                       width=600, height=440, scroll_to_selection=True)
                       #autosize_mode='fit_columns', header_row=False,


def table_callback(attrname, old, new):
    if button.label == '► Play': # table selections should only update hour, label, slider, etc. when we are in pause mode 
        selectionIndex=source_tab.selected.indices[0]
        hour, epoch = events.tolist()[selectionIndex][:2]
#         print(f'you have selected row {selectionIndex} hour {hour} epoch {epoch}={int(datetime.timestamp(t[hour].utc_datetime()))}?')    
        if hour >= hours:
            hour = 0
        source.data  = data[hour]
        source2.data = data2[hour]

        assert isinstance(slider.value, (int,float)), 'slider value should be an int or float' # value should come back as an integer numeric timestamp, specifically, the number of milliseconds since epoch
        slider.value = int(datetime.timestamp(t[hour].utc_datetime())*1000)
        slider.title = f'{datetime.fromtimestamp(slider.value/1000).date():%-d %b %Y}'
        label.text   = f'{datetime.fromtimestamp(slider.value/1000).date():%-d %b %Y}'

source_tab.selected.on_change('indices', table_callback)


df_color.columns = ['color']


plot = figure(title=f'Solar System from {t[0].utc_datetime():%d %b %Y} to {t[-1].utc_datetime():%d %b %Y}',
              tools='pan,wheel_zoom,zoom_in,zoom_out,save,reset',
              active_scroll='wheel_zoom',
              active_drag='pan',
              plot_width=510, plot_height=510)
plot.xaxis.ticker = plot.yaxis.ticker = SingleIntervalTicker(interval=1)
plot.xaxis.axis_label = 'distance in AU'
plot.yaxis.axis_label = ''
plot.title.align = 'center'
plot.title.offset = .6
plot.title.text_font_size = '12pt'
plot.title.text_font = "times"
plot.title.text_font_style = "italic"
plot.toolbar_sticky = False
plot.min_border = 40

label = Label(x=0, y=0, x_units='screen', y_units='screen',#'data'
              text_align='right', background_fill_color='black', background_fill_alpha=1,
              text=f'{t[0].utc_datetime():%-d %b %Y} '
              )

plot.add_layout(label, place='right')


color_mapper = CategoricalColorMapper(palette=df_color.color,
                                      factors=list(df_color.index.values))

# multiple line-of-sight rays from earth
plot.ray(x='earth_x', y='earth_y', angle='angle', length=12.5, angle_units='rad',
         line_width=1,
         line_color='color',
         line_dash='dashed', # 'solid', 'dotted', 'dotdash', 'dashdot',
         source=source2)

# plot the sun and planets
r = plot.circle(x='x', y='y', size='size', source=source,
    fill_color={'field': 'Body', 'transform': color_mapper},
    fill_alpha=0.8,
    line_color='#7c7e71',
    line_width=0.5,
    line_alpha=0.5,
    name='planets'
)


# some background stars from https://github.com/zingale/astro_animations/blob/master/solar_system_motion/retrograde/retrograde.py
import random
N = 10
xpos = []
ypos = []
# starbox = [hw-0.5, hw]

for s in range(N):
    # right
    xpos.append(random.uniform(xmax-0.5, xmax))
    ypos.append(random.uniform(-ymax, ymax))

    # top
    xpos.append(random.uniform(-xmax, xmax))
    ypos.append(random.uniform(ymax-0.5, ymax))

    # left
    xpos.append(random.uniform(-xmax+0.5, -xmax))#-starbox[0],-starbox[1]))
    ypos.append(random.uniform(-ymax, ymax))#-starbox[1], starbox[1]))

    # bottom
    xpos.append(random.uniform(-xmax, xmax))#-starbox[1], starbox[1]))
    ypos.append(random.uniform(-ymax+0.5, -ymax))#-starbox[0],-starbox[1]))

plot.scatter(xpos, ypos, size=10, marker='asterisk', line_color='azure')


plot.add_layout(Legend(items=[
    LegendItem(label='Sun',     renderers=[r], index=0),
    LegendItem(label='Mercury', renderers=[r], index=1),
    LegendItem(label='Venus',   renderers=[r], index=2),
    LegendItem(label='Earth',   renderers=[r], index=3),
    LegendItem(label='Mars',    renderers=[r], index=4),
    LegendItem(label='Jupiter', renderers=[r], index=5),
    LegendItem(label='Saturn',  renderers=[r], index=6),
]))

plot.legend.background_fill_color = 'black'
plot.legend.border_line_color = None
plot.legend.label_text_color = 'white'

tooltips='''
        <HTML>
        <HEAD>
        <style>
        .bk-tooltip {
            background-color: black !important;
            }
        </style>
        </HEAD>
        <BODY>
        <div>
            <div>
                <span style="color:white; font-size: 12px; ">@Body</span>
            </div>
        </div>
        </BODY>
        </HTML>'''
plot.add_tools(HoverTool(tooltips=tooltips,
                         show_arrow=False,
                         point_policy='follow_mouse',
                         names=['planets'],
                        ))


plot.background_fill_color = 'black'
plot.xaxis.minor_tick_line_color = plot.yaxis.minor_tick_line_color = None
plot.grid.grid_line_color = None
plot.xaxis.major_label_overrides = plot.yaxis.major_label_overrides = {-8: '8', -7: '7', -6: '6', -5: '5', -4: '4', -3: '3', -2: '2', -1: '1'}
plot.toolbar.autohide = True
plot.toolbar.logo=None
plot.toolbar_location='below'
plot.x_range=Range1d(-xmax, xmax, bounds=(-xmax, xmax)) # bounds keeps the plot from scrolling out past the initial position. you can scroll in but not out past the starting point. 
plot.y_range=Range1d(-ymax, ymax, bounds=(-ymax, ymax))


def animate_update():
    assert isinstance(slider.value, (int,float)), 'slider value should be an int or float' # value should come back as an integer numeric timestamp, specifically, the number of milliseconds since epoch
    hour = (datetime.fromtimestamp(86400+slider.value/1000).date()-t[0].utc_datetime().date()).days  
    if hour < 0 or hour+1 >= hours:
        hour = -1 # the `hour+1` in the following line makes this work
    slider.value = int(datetime.timestamp(t[hour+1].utc_datetime())*1000) # slightly hacky +1


num_events = len(events)
def slider_update(attrname, old, new):
    assert isinstance(slider.value, (int,float)), 'slider value should be an int or float' # value should come back as an integer numeric timestamp, specifically, the number of milliseconds since epoch
    hour = (datetime.fromtimestamp(slider.value/1000).date()-t[0].utc_datetime().date()).days
    if hour < 0 or hour >= hours:
        hour = 0
    slider.title = f'{datetime.fromtimestamp(slider.value/1000).date():%-d %b %Y}'
    label.text   = f'{datetime.fromtimestamp(slider.value/1000).date():%-d %b %Y} '
    source.data  = data[hour]
    source2.data = data2[hour]

    num = len(events[events['Epoch']<slider.value/1000]) # slider.value is in milliseconds
    if num < num_events:
        num += 1
    next_hour = pd.DataFrame(events[:num])['Frame'].tolist()[-1]
    new_source_tab = ColumnDataSource(pd.DataFrame(events[events['Frame']<=next_hour]))
    source_tab.data.update(new_source_tab.data)


slider = DateSlider(#title='Date',
                    title=f'{t[0].utc_datetime():%-d %b %Y}',
                    start=t[0].utc_datetime().date(), end=t[-1].utc_datetime().date(),
                    value=t[0].utc_datetime().date(), show_value=False,
                    width=420, height=40)#, step = 30)
slider.on_change('value', slider_update)


callback_id = None

def animate():
    global callback_id
    if button.label == '► Play':
        button.label = '❚❚ Pause'
        callback_id = curdoc().add_periodic_callback(animate_update, 100)
    else:
        button.label = '► Play'
        curdoc().remove_periodic_callback(callback_id)

button = Button(label='► Play', width=60, height = 40)
button.on_click(animate)

Layout = row([plot, column([slider, row([button, Div(text='', width=21), data_table])])],
             width = 600, height = 260)

curdoc().add_root(Layout)
curdoc().title = f'Solar System Animation'


show(plot) # shows a static plot in jupyter notebook. you need the bokeh server to make the button, slider and datatable work

