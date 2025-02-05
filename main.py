import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
from joblib import dump, load
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox


def get_height(image_path):
    cnn_model = tf.keras.models.load_model("cnn_model.h5", compile=False)
    cnn = tf.keras.Model(
        inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten").output
    )
    sample_img = Image.open(image_path)
    if sample_img.size != (128, 256):
        messagebox.showerror("Error", "Image size must be 128x256")
        return None, None

    sample_img = np.array(sample_img) / 255.0
    sample_img = tf.convert_to_tensor(sample_img, dtype=tf.float32)
    intermediate_value = cnn.predict(sample_img[None]).flatten()

    svm_model = load("best_model.joblib")
    c_0 = svm_model.predict(intermediate_value[None]).flatten()[0]
    c_1 = svm_model.predict(intermediate_value[None]).flatten()[1]
    dnn_model = tf.keras.models.load_model("best_model.h5", compile=False)
    height = (
        dnn_model.predict(
            [
                tf.convert_to_tensor(c_1, dtype=tf.float32)[None],
                tf.convert_to_tensor(c_0, dtype=tf.float32)[None],
            ]
        )
        * 100
    )
    return c_1, height


def upload_image():
    # Open file dialog and get the image file location
    file_path = filedialog.askopenfilename()
    if not file_path:
        return  # User canceled the operation

    # Open the image file with PIL
    img = Image.open(file_path)
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference to the image
    image_label.pack()

    # Simulate the function to get gender and height
    sex, height = get_height(file_path)  # Assume this function returns gender and height
    if height is not None:
        sex_str = "Male" if sex == 1 else "Female"
        message.config(text=f"The predicted height for this {sex_str} is: {height[0][0]:.2f}cm")  # Update message
        message.pack()


if __name__ == "__main__":
    # Set up the main application window
    root = tk.Tk()
    root.title("Footprint Height Prediction System")
    root.geometry("600x600")
    root.resizable(False, False)
    # Add title and version information in the title frame
    title_frame = tk.Frame(root, height=50, bg="#546e7a")
    title_frame.pack(side="top", fill="x")
    title_label = tk.Label(
        title_frame,
        text="Footprint Height Prediction Application",
        font=("Arial", 20),
        fg="#ffffff",
        bg="#546e7a",
    )
    version_label = tk.Label(
        title_frame, text="v1.0", font=("Arial", 12), fg="#ffffff", bg="#546e7a"
    )
    title_label.pack(side="left", padx=20)
    version_label.pack(side="left")
    upload_button = tk.Button(
        root,
        text="Upload Footprint Image",
        command=upload_image,
        height=3,
        width=20,
        bg="#546e7a",
        foreground="white",
        font=("helvetica", 12, "bold"),
    )
    upload_button.pack(pady=30)
    style = ttk.Style()

    # Set button style
    style.configure(
        "TButton",
        foreground="#ffffff",
        background="#546e7a",
        font=("Arial", 12),
        padding=10,
        relief="flat",
    )
    # Set text box style
    style.configure(
        "TEntry",
        foreground="#000000",
        background="#e0e0e0",
        font=("Arial", 12),
        padding=10,
    )

    image_label = tk.Label(root)
    image_label.pack()

    message = tk.Label(root, font=("Arial", 20), fg="#000000", bg="#f0f0f0")
    message.pack()
    root.mainloop()
