import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2

# Define the paths
model_weights_path = "best.pt"

# Define Streamlit app layout
layout_image = Image.open("on_shelf_availability_k.jpg")
st.image(layout_image, use_column_width=True)

# Title and description
st.title("Shelf Availability Prediction")
st.write("This app predicts shelf availability in the uploaded image.")

# Upload image for prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform prediction when a button is clicked
    if st.button("Predict"):
        # Load the YOLO model
        model = YOLO(model_weights_path)

        # Perform prediction
        detections = model.predict(image)

        # Calculate the total area of all available and not available spaces
        total_area_space_na = 0
        total_area_space_a = 0
        image_width, image_height = image.size

        # Create an ImageDraw object
        draw = ImageDraw.Draw(image)

        # Define the thickness and color of the bounding boxes
        box_thickness = 3  # Adjust this value to make the boxes thicker
        box_color = "red"  # Change this color to your preference

        for d in detections:
            boxes = d.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = int(box.cls)
                box_area = (b[2] - b[0]) * (b[3] - b[1])

                if c == 0:  # space_a
                    total_area_space_a += box_area
                    label = "space_a"
                elif c == 1:  # space_na
                    total_area_space_na += box_area
                    label = "space_na"

                # Draw  bounding boxes on the image
                for i in range(box_thickness):
                    draw.rectangle([b[0] - i, b[1] - i, b[2] + i, b[3] + i], outline=box_color)

                # Calculate label position
                label_x = (b[0] + b[2]) / 2
                label_y = (b[1] + b[3]) / 2

                # Create a background rectangle for the label
                label_bg_width = len(label) * 10  # Adjust the width based on label length
                label_bg_height = 20  # Adjust the height as needed
                label_bg_coords = (
                label_x - label_bg_width / 2, label_y - label_bg_height / 2, label_x + label_bg_width / 2,
                label_y + label_bg_height / 2)
                draw.rectangle(label_bg_coords, fill="black")

                # Draw the label
                label_font = ImageFont.truetype("arial.ttf", 12)  # Adjust the font size as needed
                text_width, text_height = draw.textsize(label, font=label_font)
                draw.text((label_x - text_width / 2, label_y - text_height / 2), label, fill="white", font=label_font)

        # Calculate available area percentage
        total_available_area = total_area_space_a / (image_width * image_height)
        available_area_percentage = total_available_area * 100

        # Calculate non-available area percentage
        non_available_area_percentage = 100 - available_area_percentage

        # Display the image with bounding boxes
        st.image(image, caption="Image with Bounding Boxes", use_column_width=True)

        # Display both available and non-available area percentages
        st.write(f"Available Space Area: {available_area_percentage:.2f}%")
        st.write(f"Non-Available Space Area: {non_available_area_percentage:.2f}%")