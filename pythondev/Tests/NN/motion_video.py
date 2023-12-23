file_name = 'C:\\Users\yair.deitcher\Downloads\\11300_CAM_vid_1688647089.avi'
import matplotlib; matplotlib.use('Qt5Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from tqdm import tqdm
from scipy.signal import resample

# Open the AVI video file (replace 'input_video.avi' with your video file)
cap = cv2.VideoCapture(file_name)

# Specify the desired frame rate (e.g., 10 frames per second)
desired_frame_rate = 1

# Define a downscale factor for frame resizing
downscale_factor = 0.5  # Adjust as needed (0.5 means frames will be half the original size)

# Initialize variables
prev_frame = None
motion_scores = []
timestamps = []
frame_count = 0

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_skip = frame_rate // desired_frame_rate  # Number of frames to skip

# Create a progress bar for video processing
pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc='Processing Frames')
frames = []
while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Skip frames if necessary to achieve the desired frame rate
    if frame_count % frame_skip != 0:
        pbar.update(1)  # Update the progress bar
        continue

    # Resize the frame to reduce processing time
    frame = cv2.resize(frame, None, fx=downscale_factor, fy=downscale_factor)
    frames.append(frame)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the current frame and the previous frame
    if prev_frame is not None:
        frame_diff = cv2.absdiff(frame_gray, prev_frame)

        # Calculate a motion score for the frame (e.g., sum of pixel differences)
        motion_score = frame_diff.sum()
        motion_scores.append(motion_score)

        # Calculate the timestamp (in minutes)
        timestamp = len(motion_scores) / desired_frame_rate / 60  # Convert seconds to minutes
        timestamps.append(timestamp)

    # Update the previous frame
    prev_frame = frame_gray.copy()

    pbar.update(1)  # Update the progress bar

# Close the progress bar
pbar.close()

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

import cmath

complex_data = np.load("C:\\Users\yair.deitcher\Downloads\\11300_NES_rawData1688647085897_10_cpx.npy",allow_pickle=True)
len_comp = complex_data[0].shape[0]
complex_real = np.real(complex_data[0][:, 8])
# complex_real = np.angle(complex_data[0][:, 8])
abs_diff_real = np.abs(np.diff(complex_real))

# Calculate timestamps for the complex data (assuming 500Hz)
complex_timestamps = np.arange(0, len(abs_diff_real) / 500, 1 / 500)

# Downsample the abs_diff_real array to 1Hz using averaging
# downsampled_abs_diff_real = resample(abs_diff_real, int(len(abs_diff_real) / 500))
downsampled_abs_diff_real = resample(abs_diff_real, len(timestamps))

# Create a figure with subplots for the video and motion score plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between subplots

# Plot the video frame with motion in the top subplot
def update(val):
    frame_idx = int(frame_slider.val)
    ax1.clear()
    ax1.imshow(cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Frame at Time: {timestamps[frame_idx]:.2f} minutes')
    ax2.clear()
    ax2.plot(timestamps, motion_scores, label='Motion Score')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Motion Score')
    ax2.set_title(f'Motion Score Over Time')
    ax2.grid(True)
    ax2.axvline(x=timestamps[frame_idx], color='r', linestyle='--')  # Add the red vertical line
    ax3.clear()
    ax3.plot(downsampled_complex_timestamps, downsampled_abs_diff_real, label='Abs Diff (Real)')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Absolute Difference')
    ax3.set_title(f'Absolute Difference of Real Part Over Time (1Hz)')
    ax3.grid(True)
    ax3.axvline(x=downsampled_complex_timestamps[frame_idx], color='r', linestyle='--')  # Add the red vertical line


# Create a downsampled timestamp array for the complex data (assuming 500Hz to 1Hz)
downsampled_complex_timestamps = np.arange(0, len(abs_diff_real) / 500, 1 / 500)[::598]
# downsampled_complex_timestamps = resample(np.arange(0, len(abs_diff_real) / 500, 1 / 500), len(timestamps))

# Plot the motion score in the middle subplot for the initial frame
ax2.plot(timestamps, motion_scores, label='Motion Score')
ax2.set_xlabel('Time (minutes)')
ax2.set_ylabel('Motion Score')
ax2.set_title(f'Motion Score Over Time')
ax2.grid(True)
ax2.axvline(x=timestamps[0], color='r', linestyle='--')  # Add the initial red vertical line

# Plot the initial absolute difference in the bottom subplot
ax3.plot(downsampled_complex_timestamps, downsampled_abs_diff_real, label='Abs Diff (Real)')
ax3.set_xlabel('Time (minutes)')
ax3.set_ylabel('Absolute Difference')
ax3.set_title(f'Absolute Difference of Real Part Over Time (1Hz)')
ax3.grid(True)
ax3.axvline(x=downsampled_complex_timestamps[0], color='r', linestyle='--')  # Add the initial red vertical line

# Add a slider for frame navigation
axframe = plt.axes([0.1, 0.02, 0.65, 0.03])
frame_slider = Slider(axframe, 'Frame', 0, len(timestamps) - 1, valinit=0, valstep=1)
frame_slider.on_changed(update)