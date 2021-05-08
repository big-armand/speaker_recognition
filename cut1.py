# -*- coding: utf-8 -*-
"""
Created on Sun May  2 13:22:58 2021

@author: edoua
"""

import ffmpeg
import os
import subprocess
import wave
#from sys import argv
import librosa

""" split_wav 'audio file' 'time listing'

    'audio file' is any file known by local FFmpeg
    'time listing' is a file containing multiple lines of format:
        'start time' 'end time' output name

    times can be either MM:SS or S*
"""

_in_file = "test1-16k.wav" #argv[1]


def make_time(elem):
    # allow user to enter times on CLI
    t = elem.split(':')
    try:
        # will fail if no ':' in time, otherwise add together for total seconds
        return int(t[0]) * 60 + float(t[1])
    except IndexError:
        return float(t[0])

def create_timing():
    os.remove("data.txt")
    fichier = open("data.txt", "a")
    i = 1
    while i < 500 :
        fichier.write(str(i-1) + " " + str(i) + " " + "sample1_" + str(i) + ".wav")
        if i < 499 :
            fichier.write("\n");
        i += 1
    fichier.close()

def collect_from_file():
    """user can save times in a file, with start and end time on a line"""

    time_pairs = []
    with open("data.txt") as in_times:
        for l, line in enumerate(in_times):
            tp = line.split()
            tp[0] = make_time(tp[0])
            tp[1] = make_time(tp[1])# - tp[0]
            # if no name given, append line count
            if len(tp) < 3:
                tp.append(str(l) + '.wav')
            time_pairs.append(tp)
    return time_pairs

def main():
    create_timing()
    cmd_string = 'ffmpeg -i {tr} -acodec copy -ss {st} -to {en} {nm}'
    for i, tp in enumerate(collect_from_file()):
        command = cmd_string.format(tr=_in_file, st=tp[0], en=tp[1], nm=tp[2])
        subprocess.call(command, shell=True)


if __name__ == '__main__':
    main()
