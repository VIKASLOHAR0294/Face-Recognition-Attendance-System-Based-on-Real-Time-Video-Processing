import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys

# Path to the folder where images of registered users will be saved
IMAGE_PATH = "ImagesAttendance"
if not os.path.exists(IMAGE_PATH):
    os.makedirs(IMAGE_PATH)

# Attendance CSV file
ATTENDANCE_FILE = "Attendance.csv"

# Function to load images and encode faces
def load_known_faces():
    images = []
    classNames = []
    encodeList = []
    for filename in os.listdir(IMAGE_PATH):
        img = cv2.imread(os.path.join(IMAGE_PATH, filename))
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])
    if images:
        encodeList = [face_recognition.face_encodings(img)[0] for img in images if len(face_recognition.face_encodings(img)) > 0]
    return encodeList, classNames

# Function to mark attendance
def mark_attendance(name):
    now = datetime.now()
    date_string = now.strftime("%Y-%m-%d")
    time_string = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{date_string},{time_string}\n")

# Check if attendance for the user has already been marked
def is_attendance_marked(name):
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(name):
                    return True
    return False

# Register a new user
def register_user(new_name, new_enrollment, frame):
    if new_name.strip() == "" or new_enrollment.strip() == "":
        return "Name and Enrollment number cannot be empty."
    new_filename = os.path.join(IMAGE_PATH, f"{new_name}_{new_enrollment}.jpg")
    cv2.imwrite(new_filename, frame)
    # Reload encodings
    global knownEncodings, knownNames
    knownEncodings, knownNames = load_known_faces()
    return f"User '{new_name}' with Enrollment '{new_enrollment}' registered successfully!"

# Function to clear attendance data and terminate the program
def clear_attendance_and_exit():
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("")  # Clear the contents of the file
    print("Attendance data cleared!")
    root.quit()  # This will close the tkinter window and terminate the program

# GUI Setup
def main_gui():
    # Initialize known encodings and names
    global knownEncodings, knownNames
    knownEncodings, knownNames = load_known_faces()

    # Function to update the video feed
    def update_video():
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(knownEncodings, face_encoding)
                face_distances = face_recognition.face_distance(knownEncodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                else:
                    best_match_index = None

                # Check if the face is recognized
                if best_match_index is not None and matches[best_match_index]:
                    name = knownNames[best_match_index].upper()
                    # Check if attendance is already marked
                    if not is_attendance_marked(name):
                        mark_attendance(name)
                        attendance_status.set(f"Attendance marked for {name}")
                    else:
                        attendance_status.set(f"Attendance already marked for {name}")
                else:
                    name = "Not Registered"

                # Draw rectangle around face
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Convert frame to ImageTk
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            video_label.imgtk = img
            video_label.configure(image=img)
        video_label.after(10, update_video)

    # Function to handle registration
    def handle_register():
        new_name = name_entry.get()
        new_enrollment = enrollment_entry.get()
        ret, frame = cap.read()
        if ret:
            message = register_user(new_name, new_enrollment, frame)
            registration_status.set(message)

    # Main Window
    global root
    root = tk.Tk()
    root.title("Face Recognition Attendance System")

    # Left Panel (Video Feed)
    video_frame = ttk.Frame(root)
    video_frame.grid(row=0, column=0, padx=10, pady=10)
    video_label = ttk.Label(video_frame)
    video_label.grid(row=0, column=0)

    # Right Panel (Registration)
    register_frame = ttk.Frame(root)
    register_frame.grid(row=0, column=1, padx=10, pady=10)
    ttk.Label(register_frame, text="New User Registration").grid(row=0, column=0, columnspan=2, pady=5)
    ttk.Label(register_frame, text="Name:").grid(row=1, column=0, padx=5, pady=5)
    name_entry = ttk.Entry(register_frame)
    name_entry.grid(row=1, column=1, padx=5, pady=5)
    ttk.Label(register_frame, text="Enrollment No:").grid(row=2, column=0, padx=5, pady=5)
    enrollment_entry = ttk.Entry(register_frame)
    enrollment_entry.grid(row=2, column=1, padx=5, pady=5)
    register_button = ttk.Button(register_frame, text="Register", command=handle_register)
    register_button.grid(row=3, column=0, columnspan=2, pady=10)

    # Status Labels
    attendance_status = tk.StringVar()
    ttk.Label(root, textvariable=attendance_status).grid(row=1, column=0, columnspan=2, pady=10)
    registration_status = tk.StringVar()
    ttk.Label(root, textvariable=registration_status).grid(row=2, column=0, columnspan=2, pady=10)

    # Start Video Feed
    global cap
    cap = cv2.VideoCapture(0)
    update_video()

    # Key binding for clearing attendance data and terminating the program (Delete key)
    root.bind("<Delete>", lambda event: clear_attendance_and_exit())

    # Run the GUI
    root.mainloop()

    # Release the camera when the GUI is closed
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_gui()
