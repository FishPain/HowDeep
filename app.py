from flask import Flask, render_template, request
from src.ml.inference_worker import InferenceWorker
import os

from src.app import app_utils
from io import BytesIO

PORT = int(os.getenv("PORT", 5001))
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Helper function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def root():
    processed_image = None
    error_message = None

    if request.method == "POST":
        if "image" not in request.files:
            error_message = "No file part"
            return render_template("home.html", error_message=error_message)

        file = request.files["image"]
        if file.filename == "":
            error_message = "No selected file"
            return render_template("home.html", error_message=error_message)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                # Process the image
                inference_worker = InferenceWorker(filepath)
                processed_image = (
                    inference_worker.process_image()
                )  # Assumes a method `process_image`

                # Save the processed image to a buffer for returning
                img_io = BytesIO()
                processed_image.save(img_io, format="PNG")
                img_io.seek(0)

                return send_file(img_io, mimetype="image/png")

            except Exception as e:
                error_message = f"Error during image processing: {e}"
        else:
            error_message = "Invalid file type. Please upload a PNG or JPG file."

    return render_template("home.html", error_message=error_message)


# 404 Page
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=PORT)
