from PIL import Image, ImageTk
import cv2
import pytesseract
import tkinter as tk
import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load environment variables from .env file
load_dotenv()

# Access Mistral AI API Key
api_key = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=api_key)

# Choose a Mistral AI model engine
model_engine = "mistral-large-latest"

# Configure Tesseract
# To check language abbreviations: pytesseract.get_languages(config='.')
config = "-l eng+jpn+kor+chi_sim+thai+hin"


def load_image(file_name):

    image = cv2.imread(file_name)

    return image


def format_image(image):

    # Rescale image
    width = 350
    height = 250
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite('Temp/Resized.jpg', resized)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find Image Contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour
    for contour in contours:
        # Determine the location and size of the object
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle on the detected object
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite('Temp/Contour.jpg', image)

    return image, threshold, dim


def extract_text(threshold):

    # Extraction of text from images
    text = pytesseract.image_to_string(threshold, config=config)

    return text


def text_translation(text):

    chat_response = client.chat(
        model=model_engine,
        temperature=0,
        messages=[ChatMessage(
            role="user",
            content="Translate "+text+" to English"),
            ChatMessage(
                role="system",
                content="You are a language translation expert. Translate the given text accurately in English. Do not explain.")
        ]
    )

    # Get Translation Results
    translated_text = chat_response.choices[0].message.content.strip()

    return translated_text


def main():

    # Load Image
    input_value = input("Enter Image Name (Only PNG, JPG, JPEG accepted)\n")
    image = load_image('Input/'+input_value)

    # Format Image and detect contours
    formatted_image, threshold, dimension = format_image(image)

    # Extract Text from Image
    extracted_text = extract_text(threshold)

    # Translate extracted text into English
    translated_text = text_translation(extracted_text)

    # Show Detection Results
    root = tk.Tk()

    # Original Image
    original_image = Image.open('Temp/Resized.jpg')
    Image1 = ImageTk.PhotoImage(original_image)

    # Segmented Image
    contour = cv2.imread('Temp/Contour.jpg')
    resized_contour = cv2.resize(contour, dimension, interpolation=cv2.INTER_AREA)
    cv2.imwrite('Temp/Contour-resized.jpg', resized_contour)
    img_contour = Image.open('Temp/Contour-resized.jpg')
    Image2 = ImageTk.PhotoImage(img_contour)

    # Format GUI
    root.title("Text Detection and Language Translation")

    # Detection and Translation
    image_origin = tk.Label(root, image=Image1)
    image_origin.grid(row=0, column=0)
    image_segments = tk.Label(root, image=Image2)
    image_segments.grid(row=0, column=1)

    # Show detected text
    label_text_detected = tk.Label(root, text="Detected Text : \n" + extracted_text)
    label_text_detected.grid(row=1, column=0)

    # Defines the font size and window size
    font_size = 10
    window_width = 800
    characters_per_line = window_width // font_size

    # Get the number of characters that can be displayed on one line
    text_lines = [translated_text[i:i + characters_per_line] for i in range(0, len(translated_text), characters_per_line)]

    # Display text with tkinter
    label_text_translate = tk.Label(root, text="Translated Text : \n" + "\n".join(text_lines), font=("Helvetica", font_size))
    label_text_translate.grid(row=1, column=1)

    # Run GUI
    root.mainloop()


if __name__ == "__main__":
    main()