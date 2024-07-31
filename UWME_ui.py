import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import moviepy.editor as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter import Toplevel, Label
from threading import Thread
from PIL import Image, ImageTk
import time

# Function to split video into frames
def split_video_into_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open the video file: {video_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

# Function to merge frames into a video
def merge_frames_into_video(frames, output_path, codec, fps, frame_width, frame_height):
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter('temp_video.mp4', fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        print(f"Error: Could not open the video writer: {output_path}")
        return False

    for i, frame in enumerate(frames):
        out.write(frame)
        print(f"Writing frame {i + 1} of {len(frames)}")

    out.release()
    return True

def should_return_dehazed_only(enhance_type):
    return enhance_type == 'dehazed_only'

# Function to enhance a single frame
def enhance_frame(img, alpha, beta, hue_shift, top_percent, omega, enhance_type):
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

    def estimate_atmospheric_light(img, dark_channel, top_percent):
        num_pixels = np.prod(img.shape[:2])
        num_brightest = int(num_pixels * top_percent)
        indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
        brightest_pixels = img.reshape(-1, 3)[indices]
        atmospheric_light = np.mean(brightest_pixels, axis=0)
        return atmospheric_light

    def transmission_map(img, atmos_light, omega):
        normalized_img = img.astype(np.float32) / atmos_light
        transmission = 1 - omega * np.min(normalized_img, axis=2)
        return transmission

    def refine_transmission_map(img, transmission, radius=60, eps=0.0001):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        refined_transmission = guided_filter(gray_img, transmission, radius, eps)
        return refined_transmission

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

    def recover_scene_radiance(img, transmission, atmos_light, t0=0.1):
        transmission = np.maximum(transmission, t0)
        J = np.empty_like(img)
        for i in range(3):
            J[:, :, i] = (img[:, :, i] - atmos_light[i]) / transmission + atmos_light[i]
        J = np.clip(J, 0, 255).astype(np.uint8)
        return J

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

    dark_channel = dcp(img)
    atmospheric_light = estimate_atmospheric_light(img, dark_channel, top_percent)
    raw_transmission = transmission_map(img, atmospheric_light, omega)
    refined_transmission = refine_transmission_map(img, raw_transmission)
    dehazed_img = recover_scene_radiance(img, refined_transmission, atmospheric_light)
    soft_refined_transmission = soft_matting(img, refined_transmission)
    dehazed_img_soft_Matting = recover_scene_radiance(img, soft_refined_transmission, atmospheric_light)
    
    final_dehazed_image = dehazed_img_soft_Matting
    
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
    
    if should_return_dehazed_only(enhance_type):
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
    update_progress(25)
    time.sleep(1)  # Simulating some processing time 
    update_progress(50)
    time.sleep(1)
    update_progress(75)
    time.sleep(1) 
    update_progress(100)
    print(f"Image successfully written to {output_image_path}")

def process_video(input_video_path, output_video_path, alpha, beta, hue_shift, top_percent, omega, enhance_type):
    video_clip = mp.VideoFileClip(input_video_path)
    audio_clip = video_clip.audio
    frames = split_video_into_frames(input_video_path)
    if not frames:
        return

    total_frames = len(frames)
    enhanced_frames = []
    for i, frame in enumerate(frames):
        enhanced_frame = enhance_frame(frame, alpha, beta, hue_shift, top_percent, omega, enhance_type)
        enhanced_frames.append(enhanced_frame)
        progress = int((i + 1) / total_frames * 100)
        update_progress(progress)

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

# Tkinter UI for file selection and enhancement type
def browse_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        input_path_var.set(file_path)

def save_file():
    file_extension = os.path.splitext(input_path_var.get())[1].lower()
    if file_extension in ['.jpg', '.jpeg', '.png']:
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    elif file_extension in ['.mp4', '.avi', '.mov']:
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                                 filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("MOV files", "*.mov"), ("All files", "*.*")])
    else:
        messagebox.showerror("Error", "Unsupported file format")
        return

    if file_path:
        output_path_var.set(file_path)

def show_processing_window():
    global processing_window, progress_label, loading_label
    processing_window = Toplevel(root)
    processing_window.title("Processing...")
    processing_window.geometry("350x80")  # Adjust size as needed

    # Loading GIF
    global loading_gif
    loading_gif = ImageTk.PhotoImage(Image.open("loading.gif").resize((30, 30)))
    loading_label = Label(processing_window, image=loading_gif)
    loading_label.grid(row=0, column=0, padx=10, pady=10)

    # Progress label (combined text and percentage)
    progress_label = Label(processing_window, text="This might take a while... 0%")
    progress_label.grid(row=0, column=1, padx=10, pady=10)

    processing_window.resizable(False, False) 
    processing_window.transient(root)

    processing_window.grab_set()
    root.update()

def hide_processing_window():
    global processing_window
    if processing_window:
        processing_window.destroy()
        processing_window = None
    root.update()

def update_progress(percent):
    global progress_label
    progress_label.config(text=f"This might take a while... {percent}% Don't Close This Window")
    root.update()

def apply_changes():
    input_path = input_path_var.get()
    output_path = output_path_var.get()
    alpha = float(alpha_var.get())
    beta = float(beta_var.get())
    hue_shift = float(hue_var.get())
    top_percent = float(top_var.get())
    omega = float(omega_var.get())
    enhance_type = enhance_type_var.get()

    if not input_path or not output_path:
        messagebox.showerror("Error", "Please select both input and output files")
        return

    show_processing_window()

    processing_thread = Thread(target=main, args=(input_path, output_path, alpha, beta, hue_shift, top_percent, omega, enhance_type))
    processing_thread.start()

    while processing_thread.is_alive():
        root.update()
    hide_processing_window()

    messagebox.showinfo("Success", "Processing complete!")

# Create the Tkinter window
root = tk.Tk()
root.title("Underwater Image/Video Enhancer")

input_path_var = tk.StringVar()
output_path_var = tk.StringVar()
alpha_var = tk.StringVar(value="0.8")
beta_var = tk.StringVar(value="0")
hue_var = tk.StringVar(value="5.8")
top_var = tk.StringVar(value="0.001")
omega_var = tk.StringVar(value="0.6")
enhance_type_var = tk.StringVar(value="normal")

processing_window = None 

# Layout
tk.Label(root, text="Select Input File:").grid(row=0, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=input_path_var, width=50).grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Select Output File:").grid(row=1, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=output_path_var, width=50).grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Save As", command=save_file).grid(row=1, column=2, padx=10, pady=5)

tk.Label(root, text="Alpha:").grid(row=2, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=alpha_var).grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Beta:").grid(row=3, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=beta_var).grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Hue Shift:").grid(row=4, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=hue_var).grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Top Percent:").grid(row=5, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=top_var).grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Omega:").grid(row=6, column=0, padx=10, pady=5)
tk.Entry(root, textvariable=omega_var).grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Enhance Type:").grid(row=7, column=0, padx=10, pady=5)
ttk.Radiobutton(root, text="Normal", variable=enhance_type_var, value="normal").grid(row=7, column=1, padx=10, pady=5, sticky='w')
ttk.Radiobutton(root, text="Dehazed Only", variable=enhance_type_var, value="dehazed_only").grid(row=7, column=1, padx=10, pady=5)

tk.Button(root, text="Apply", command=apply_changes).grid(row=8, column=0, columnspan=3, padx=10, pady=20)

root.mainloop()