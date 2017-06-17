#!/bin/sh
# delay X would give 100/X FPS in the video for the animation

convert -delay 5 -loop 0 transfer_test__at_iteration_*.png animation.gif
