import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import moviepy.editor as mp

#Run this if you face any error regarding moviepy module not installed: "pip install moviepy"

# Function to split video into frames
def split_video_into_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file: {video_path}")
        return []

    frames = []
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # Release the video capture object
    cap.release()
    return frames

# Function to merge frames into a video
def merge_frames_into_video(frames, output_path, codec, fps, frame_width, frame_height):
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (frame_width, frame_height))  # Changed output to temp_video.mp4
    if not out.isOpened():
        print(f"Error: Could not open the video writer: {output_path}")
        return False

    for i, frame in enumerate(frames):
        out.write(frame)
        print(f"Writing frame {i + 1} of {len(frames)}")

    # Release the VideoWriter object
    out.release()
    return True

def should_return_dehazed_only(enhance_type):
    return enhance_type == 'dehazed_only'

# Function to enhance a single frame
def enhance_frame(img, alpha, beta, hue_shift, top_percent, omega, enhance_type):
    # Function to compute the Dark Channel Prior (DCP)
    def dcp(img, patch_size=15):
        dark_channel = np.zeros_like(img[:, :, 0])
        for y in range(0, img.shape[0], patch_size):
            for x in range(0, img.shape[1], patch_size):
                patch = img[max(0, y):min(y + patch_size, img.shape[0]),
                            max(0, x):min(x + patch_size, img.shape[1])]
                patch_min = np.min(patch, axis=2)
                dark_channel[max(0, y):min(y + patch_size, img.shape[0]),
                             max(0, x):min(x + patch_size, img.shape[1])] = patch_min
        return dark_channel

    # Function to estimate atmospheric light
    def estimate_atmospheric_light(img, dark_channel, top_percent):
        num_pixels = np.prod(img.shape[:2])
        num_brightest = int(num_pixels * top_percent)
        indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
        brightest_pixels = img.reshape(-1, 3)[indices]
        atmospheric_light = np.mean(brightest_pixels, axis=0)
        return atmospheric_light

    # Function to compute the transmission map
    def transmission_map(img, atmos_light, omega):
        normalized_img = img.astype(np.float32) / atmos_light
        transmission = 1 - omega * np.min(normalized_img, axis=2)
        return transmission
    
    # Function to refine the transmission map using guided filter
    def refine_transmission_map(img, transmission, radius=60, eps=0.0001):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        refined_transmission = guided_filter(gray_img, transmission, radius, eps)
        return refined_transmission

    # Guided filter function
    def guided_filter(I, p, r, eps):
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * I + mean_b
        return q

    # Function to recover the scene radiance
    def recover_scene_radiance(img, transmission, atmos_light, t0=0.1):
        transmission = np.maximum(transmission, t0)
        J = np.empty_like(img)
        for i in range(3):
            J[:, :, i] = (img[:, :, i] - atmos_light[i]) / transmission + atmos_light[i]
        J = np.clip(J, 0, 255).astype(np.uint8)
        return J

    # Soft matting function
    def soft_matting(img, transmission, radius=15, eps=1e-7):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        refined_transmission = guided_filter(gray_img, transmission, radius, eps)
        return refined_transmission
    
    def histogram_equalization(channel):
        equalized_channel = cv2.equalizeHist(channel)
        return equalized_channel

    def adjust_contrast_brightness(image, alpha, beta):
        adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted_image

    def adjust_hue(image, hue_shift):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_image)
        h = np.mod(h + hue_shift, 180).astype(np.uint8)
        adjusted_hsv_image = cv2.merge([h, s, v])
        adjusted_rgb_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
        return adjusted_rgb_image

    def process_rgb_image(image, method, *args):
        b, g, r = cv2.split(image)
        b_processed = method(b, *args)
        g_processed = method(g, *args)
        r_processed = method(r, *args)
        processed_image = cv2.merge([b_processed, g_processed, r_processed])
        return processed_image

    # Start of enhancement process
    dark_channel = dcp(img)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel, top_percent)
    raw_transmission = transmission_map(img, atmospheric_light, omega)
    refined_transmission = refine_transmission_map(img, raw_transmission)
    dehazed_img = recover_scene_radiance(img, refined_transmission, atmospheric_light)
    soft_refined_transmission = soft_matting(img, refined_transmission)
    dehazed_img_soft_Matting = recover_scene_radiance(img, soft_refined_transmission, atmospheric_light)
    
    final_dehazed_image = dehazed_img_soft_Matting

    # Apply Gaussian blur for noise reduction
    #kernel_size = 3
    #sigma = 0.6
    #blurred_img = cv2.GaussianBlur(dehazed_img_soft_Matting, (kernel_size, kernel_size), sigma)

    # Final dehazed image
    final_dehazed_image = dehazed_img_soft_Matting
    
    # White balancing the final dehazed image
    scale_r = 0.9
    scale_g = 0.8
    scale_b = 0.7
    white_balanced_image = final_dehazed_image.copy()
    white_balanced_image[:, :, 0] = final_dehazed_image[:, :, 0] * scale_r
    white_balanced_image[:, :, 1] = final_dehazed_image[:, :, 1] * scale_g
    white_balanced_image[:, :, 2] = final_dehazed_image[:, :, 2] * scale_b
    white_balanced_image = np.clip(white_balanced_image, 0, 255).astype(np.uint8)
    
    equalized_image = process_rgb_image(white_balanced_image, histogram_equalization)
    adjusted_contrast_brightness_image = adjust_contrast_brightness(equalized_image, alpha, beta)
    adjusted_image = adjust_hue(adjusted_contrast_brightness_image, hue_shift)
    
    if should_return_dehazed_only(enhance_type):  # Modified this line
        return final_dehazed_image
    else:
        return adjusted_image


# Function to process an image
def process_image(input_image_path, output_image_path, alpha, beta, hue_shift, top_percent, omega, enhance_type):
    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not read the image file: {input_image_path}")
        return

    enhanced_img = enhance_frame(img, alpha, beta, hue_shift, top_percent, omega, enhance_type)
    cv2.imwrite(output_image_path, enhanced_img)
    print(f"Image successfully written to {output_image_path}")

def process_video(input_video_path, output_video_path, alpha, beta, hue_shift, top_percent, omega, enhance_type):
    video_clip = mp.VideoFileClip(input_video_path)
    audio_clip = video_clip.audio
    frames = split_video_into_frames(input_video_path)
    if not frames:
        return

    enhanced_frames = [enhance_frame(frame, alpha, beta, hue_shift, top_percent, omega, enhance_type) for frame in frames]
    frame_height, frame_width = frames[0].shape[:2]
    fps = 30
    codec = 'XVID'

    success = merge_frames_into_video(enhanced_frames, output_video_path, codec, fps, frame_width, frame_height)
    if success:
        processed_video_clip = mp.VideoFileClip('temp_video.mp4')
        processed_video_clip = processed_video_clip.set_audio(audio_clip)
        processed_video_clip.write_videofile(output_video_path, codec='libx264')
        os.remove('temp_video.mp4')
        print(f"Video successfully written to {output_video_path}")
    else:
        print("Failed to write video")

def main(input_path, output_path, alpha, beta, hue_shift, top_percent, omega, enhance_type):
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist")
        return

    file_extension = os.path.splitext(input_path)[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        process_image(input_path, output_path, alpha, beta, hue_shift, top_percent, omega, enhance_type)
    elif file_extension in ['.mp4', '.avi', '.mov']:
        process_video(input_path, output_path, alpha, beta, hue_shift, top_percent, omega, enhance_type)
    else:
        print(f"Unsupported file format: {file_extension}")

if __name__ == "__main__":
    input_path = 'video/input_vid.mp4'
    output_path = 'video/output_vid.mp4'
    alpha = 0.8
    beta = 0
    hue_shift = 5.8
    top_percent = 0.001
    omega = 0.6
    enhance_type = 'normal'
    main(input_path, output_path, alpha, beta, hue_shift, top_percent, omega, enhance_type)