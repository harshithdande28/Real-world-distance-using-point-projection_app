from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2 as cv
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chessboard settings (7x6 corners)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# ----------------------------
# Global variables to store intrinsic parameters
# ----------------------------
calibration_done = False
Fx, Fy, Ox, Oy = None, None, None, None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/calibrate", methods=["POST"])
def calibrate():
    global calibration_done, Fx, Fy, Ox, Oy
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    objpoints, imgpoints = [], []
    gray = None

    for f in files:
        filepath = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(filepath)
        img = cv.imread(filepath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if not objpoints:
        return jsonify({"error": "No chessboard corners detected"}), 400

    # Run calibration (ignore distortion coefficients)
    ret, mtx, _, _, _ = cv.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Extract intrinsic parameters
    Fx, Fy = mtx[0, 0], mtx[1, 1]
    Ox, Oy = mtx[0, 2], mtx[1, 2]

    calibration_done = True

    return jsonify({
        "Fx": float(Fx),
        "Fy": float(Fy),
        "Ox": float(Ox),
        "Oy": float(Oy)
    })

@app.route("/get_intrinsics", methods=["GET"])
def get_intrinsics():
    if not calibration_done:
        return jsonify({"error": "Calibration not done yet"}), 400

    return jsonify({
        "Fx": float(Fx),
        "Fy": float(Fy),
        "Ox": float(Ox),
        "Oy": float(Oy)
    })

if __name__ == "__main__":
    app.run(debug=True)
