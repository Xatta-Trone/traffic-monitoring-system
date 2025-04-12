import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from collections import defaultdict, deque
import time
from math import radians, sin, cos, sqrt, atan2
import csv
from datetime import datetime
import os
import torch
import torchvision

# Automatically choose device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
print("✅ CUDA available:", torch.cuda.is_available())
print("🧠 Torch version:", torch.__version__)
print("🖼️ Torchvision version:", torchvision.__version__)
print("🖥️ CUDA Device:", torch.cuda.get_device_name(
    0) if torch.cuda.is_available() else "None")

yolo_model_name = 'yolo11m_train.pt'
video_name = "lbj_trains.mp4"
video_path = os.path.join('assets', 'small videos', video_name)
output_path = os.path.join('assets', 'small videos',
                           video_name.replace('.mp4', '_output.mp4'))





def calculate_distance_feet(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in feet using haversine formula."""
    R = 20902231  # Earth's radius in feet
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def calculate_pixel_to_feet_ratio(roi_points, point_coordinates):
    """Calculate the pixel-to-feet ratio using ROI points and their GPS coordinates."""
    pixel_distances = []
    real_distances = []

    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]

        # Calculate pixel distance
        pixel_dist = sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
        pixel_distances.append(pixel_dist)

        # Calculate real distance using GPS
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        real_dist = calculate_distance_feet(lat1, lon1, lat2, lon2)
        real_distances.append(real_dist)

    # Calculate average conversion ratio (feet per pixel)
    ratios = [real / pixel for real,
              pixel in zip(real_distances, pixel_distances)]
    return sum(ratios) / len(ratios)


def interpolate_gps(x, y, roi_points, point_coordinates):
    """Interpolate GPS coordinates for a point inside ROI (a heuristic if needed)."""
    total_area = 0
    weighted_lat = 0
    weighted_lon = 0

    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]
        pt3 = (x, y)

        # Calculate triangle area via the shoelace formula
        area = abs((pt1[0] * (pt2[1] - pt3[1]) + pt2[0] *
                    (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1])) / 2)
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        avg_lat = (lat1 + lat2) / 2
        avg_lon = (lon1 + lon2) / 2

        weighted_lat += avg_lat * area
        weighted_lon += avg_lon * area
        total_area += area

    if total_area > 0:
        return weighted_lat / total_area, weighted_lon / total_area
    return None


def draw_distances(frame, roi_points, point_coordinates):
    """Draw distances on each edge of the ROI."""
    distances = []
    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]

        # Get GPS coordinates for the points and calculate distance
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        dist = calculate_distance_feet(lat1, lon1, lat2, lon2)
        distances.append(dist)

        # Calculate midpoint for text placement
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        text = f"{dist:.1f} ft"
        cv2.putText(frame, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(frame, text, (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return distances


def get_perspective_transform(source_points, target_points):
    """Get perspective transform matrix from source to target points."""
    source = np.array(source_points, dtype=np.float32)
    target = np.array(target_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)


def transform_points(points, transform_matrix):
    """Transform a list of points using a perspective transform matrix."""
    if len(points) == 0:
        return points
    reshaped_points = np.array(points).reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(
        reshaped_points, transform_matrix)
    return transformed_points.reshape(-1, 2)


def setup_perspective_transform(roi_points):
    """Setup a bird’s-eye view perspective transform based on ROI points."""
    x_min = min(pt[0] for pt in roi_points)
    x_max = max(pt[0] for pt in roi_points)
    y_min = min(pt[1] for pt in roi_points)
    y_max = max(pt[1] for pt in roi_points)
    width = x_max - x_min
    height = y_max - y_min
    target_points = np.array([
        [x_min, y_min],
        [x_min + width, y_min],
        [x_min + width, y_min + height],
        [x_min, y_min + height]
    ], dtype=np.float32)
    return get_perspective_transform(roi_points, target_points)


def calculate_speed(prev_pos, curr_pos, time_elapsed, pixel_to_feet, transform_matrix):
    """Calculate speed (mph) using a bird's-eye view transform from pixel positions."""
    if not time_elapsed:
        return 0
    transformed_points = transform_points(
        [prev_pos, curr_pos], transform_matrix)
    prev_transformed = transformed_points[0]
    curr_transformed = transformed_points[1]
    dx = curr_transformed[0] - prev_transformed[0]
    dy = curr_transformed[1] - prev_transformed[1]
    pixel_dist = np.sqrt(dx * dx + dy * dy)
    feet_dist = pixel_dist * pixel_to_feet
    speed_mph = (feet_dist / time_elapsed) * 0.681818
    return speed_mph


def transform_pixel_to_gps(pixel_point, H):
    """
    Convert a pixel coordinate (x, y) to its corresponding GPS coordinate using the homography matrix H.
    Returns a NumPy array [lat, lon].
    """
    point_h = np.array([pixel_point[0], pixel_point[1], 1.0], dtype=np.float32)
    gps_h = H.dot(point_h)
    gps_point = gps_h[:2] / gps_h[2]
    return gps_point


def calculate_speed_gps(prev_gps, curr_gps, time_elapsed):
    """
    Calculate speed (mph) based on two GPS coordinates and the elapsed time (in seconds).
    Uses calculate_distance_feet() for distance in feet.
    """
    feet_dist = calculate_distance_feet(
        prev_gps[0], prev_gps[1], curr_gps[0], curr_gps[1])
    speed_mph = (feet_dist / time_elapsed) * 0.681818
    return speed_mph


# Global variables for ROI and line selection
roi_points = []
line_points = []  # For storing line start and end points
is_selecting_roi = True
is_selecting_line = False
point_coordinates = {}

# Dictionary to store previous GPS positions for speed calculation
previous_gps_positions = {}


def click_event(event, x, y, flags, param):
    global roi_points, line_points, is_selecting_roi, is_selecting_line
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_selecting_roi:
            coords = input(f"Enter latitude,longitude for point ({x},{y}): ")
            try:
                lat, lon = map(float, coords.split(','))
                roi_points.append((x, y))
                point_coordinates[(x, y)] = (lat, lon)
                print(f"Added ROI point: ({x},{y}) with GPS: {lat},{lon}")
                if len(roi_points) == 4:
                    is_selecting_roi = False
                    is_selecting_line = True
                    print("Now select 2 points for the counting line.")
            except ValueError:
                print("Invalid input! Please enter latitude,longitude correctly.")
                return
        elif is_selecting_line:
            line_points.append((x, y))
            print(f"Added line point: ({x},{y})")
            if len(line_points) == 2:
                is_selecting_line = False


# Load the video and get the first frame
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

cv2.namedWindow('ROI Selection')
cv2.setMouseCallback('ROI Selection', click_event)

print("Select 4 points for Vehicle ROI")
print("Enter coordinates as: latitude,longitude")

while is_selecting_roi or is_selecting_line:
    temp_frame = frame.copy()
    if len(roi_points) > 0:
        for point in roi_points:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            if point in point_coordinates:
                lat, lon = point_coordinates[point]
                cv2.putText(temp_frame, f"{lat:.6f}, {lon:.6f}", (point[0] + 10, point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(roi_points) > 1:
            cv2.polylines(temp_frame, [np.array(
                roi_points, np.int32)], True, (0, 0, 255), 2)
    if len(line_points) > 0:
        for point in line_points:
            cv2.circle(temp_frame, point, 5, (255, 0, 255), -1)
        if len(line_points) == 2:
            cv2.line(temp_frame, line_points[0],
                     line_points[1], (255, 0, 255), 2)
    if is_selecting_roi:
        cv2.putText(temp_frame, f"ROI Points: {len(roi_points)}/4",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_selecting_line:
        cv2.putText(temp_frame, f"Line Points: {len(line_points)}/2",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ROI Selection', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('ROI Selection')

# Create ROI polygon from selected points
roi_polygon = Polygon(roi_points)
line_start = line_points[0]
line_end = line_points[1]

# Save ROI coordinates for record keeping
with open('roi_coordinates.txt', 'w') as f:
    f.write("Vehicle ROI Points:\n")
    for point in roi_points:
        lat, lon = point_coordinates[point]
        f.write(f"Point {point}: {lat},{lon}\n")

# Compute the homography matrix mapping ROI pixel points to their corresponding GPS coordinates.
roi_pixel_points = np.array(roi_points, dtype=np.float32)
roi_gps_points = np.array([point_coordinates[pt]
                          for pt in roi_points], dtype=np.float32)
GPS_H, status = cv2.findHomography(roi_pixel_points, roi_gps_points)
print("GPS Homography Matrix:", GPS_H)

# Get FPS and calculate the smoothing window (half a second minimum 3 frames)
fps = int(cap.get(cv2.CAP_PROP_FPS))
SMOOTHING_WINDOW = max(3, fps // 2)

# Calculate pixel-to-feet ratio (for use in alternate speed calculations if needed)
PIXEL_TO_FEET = calculate_pixel_to_feet_ratio(roi_points, point_coordinates)

# Setup perspective transform (for bird's-eye view, if needed)
TRANSFORM_MATRIX = setup_perspective_transform(roi_points)

print(f"Video FPS: {fps}")
print(f"Calculated PIXEL_TO_FEET ratio: {PIXEL_TO_FEET:.4f}")
print(f"Using smoothing window of {SMOOTHING_WINDOW} frames")



# Load the YOLO model and move to the selected device
model = YOLO(yolo_model_name)
model.to(DEVICE)

# Get the class list for labeling
class_list = model.names

print(width, height, fps)

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

line_color = (255, 0, 255)
line_thickness = 2

# Counters, trajectory storage, and speed tracking
crossed_ids = set()
crossed_ids_down = set()
crossed_ids_up = set()
class_counts = defaultdict(int)
class_counts_direction = {
    'top_to_bottom': defaultdict(int),
    'bottom_to_top': defaultdict(int)
}
trajectories = defaultdict(lambda: deque())
active_objects = set()
object_speeds = {}
previous_positions = {}  # (for pixel-based, if still used)
speed_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
previous_gps_positions = {}  # For GPS-based speed calculation

frame_count = 0

# Create CSV file for logging tracking data
csv_filename = f"tracking_data_{os.path.splitext(video_name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Object_ID', 'Object_Type', 'Speed_MPH',
                     'Current_Lat', 'Current_Long', 'Trajectory_Points'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1

    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[
                          0, 1, 2, 3, 5, 6, 7], device=DEVICE)

    cv2.polylines(frame, [np.array(roi_points, np.int32)],
                  isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.line(frame, line_start, line_end, line_color, line_thickness)

    # Draw distances on the ROI edges
    distances = draw_distances(frame, roi_points, point_coordinates)
    if frame_count % 30 == 0:
        print("\nEdge distances:")
        for i, dist in enumerate(distances):
            print(f"Edge {i+1}: {dist:.1f} ft")
        print(f"Total perimeter: {sum(distances):.1f} ft")

    new_active_objects = set()
    frame_data = []
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if results[0].boxes is None or results[0].boxes.data is None:
        cv2.imshow('Polygon on Frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Ensure that detections exist before processing
    if (results is not None and len(results) > 0 and
        results[0].boxes is not None and
            hasattr(results[0].boxes, 'data') and results[0].boxes.data is not None and
            (hasattr(results[0].boxes, 'id') and hasattr(results[0].boxes.id, 'int') and callable(results[0].boxes.id.int))):
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]

            # Check if the object is inside the ROI
            pt = Point(cx, cy)
            if not roi_polygon.contains(pt):
                if track_id in trajectories:
                    del trajectories[track_id]
                continue

            new_active_objects.add(track_id)
            trajectories[track_id].append((cx, cy))

            # Convert current pixel center to GPS using the computed homography
            current_gps = transform_pixel_to_gps((cx, cy), GPS_H)

            # Calculate speed based on GPS coordinates if a previous GPS exists
            if track_id in previous_gps_positions:
                prev_gps, prev_time = previous_gps_positions[track_id]
                time_elapsed = current_time - prev_time
                if time_elapsed > 0:
                    current_speed = calculate_speed_gps(
                        prev_gps, current_gps, time_elapsed)
                    speed_history[track_id].append(current_speed)
                    if len(speed_history[track_id]) >= 3:
                        smoothed_speed = sum(
                            speed_history[track_id]) / len(speed_history[track_id])
                        object_speeds[track_id] = round(smoothed_speed, 1)
            else:
                current_speed = 0

            previous_gps_positions[track_id] = (current_gps, current_time)

            label = f"ID: {track_id} | {class_name} | {object_speeds.get(track_id, 0):.1f} mph"
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 30),
                          (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if track_id in trajectories:
                for i in range(1, len(trajectories[track_id])):
                    if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                        continue
                    cv2.line(
                        frame, trajectories[track_id][i - 1], trajectories[track_id][i], (255, 255, 0), 2)

            # Check if the object crosses the counting line
            if track_id not in crossed_ids_down and track_id not in crossed_ids_up:
                prev_pos = trajectories[track_id][-2] if len(
                    trajectories[track_id]) > 1 else None
                if prev_pos:
                    prev_x, prev_y = prev_pos
                    curr_x, curr_y = cx, cy
                    line_slope = (line_end[1] - line_start[1]) / \
                        (line_end[0] - line_start[0])
                    line_y = line_start[1] + line_slope * (cx - line_start[0])
                    if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                        if prev_y < curr_y:
                            crossed_ids_down.add(track_id)
                            class_counts_direction['top_to_bottom'][class_name] += 1
                            direction = "Top to Bottom"
                        else:
                            crossed_ids_up.add(track_id)
                            class_counts_direction['bottom_to_top'][class_name] += 1
                            direction = "Bottom to Top"
                        print(
                            f"Counted: {class_name}, ID: {track_id}, Direction: {direction}")
                        class_counts[class_name] += 1

            # Optionally, use interpolate_gps as a fallback for logging
            interp_gps = interpolate_gps(cx, cy, roi_points, point_coordinates)
            if interp_gps:
                frame_data.append([
                    current_timestamp,
                    track_id,
                    class_name,
                    object_speeds.get(track_id, 0),
                    interp_gps[0],
                    interp_gps[1],
                    f"({interp_gps[0]:.6f}, {interp_gps[1]:.6f})"
                ])
    # else:
    #     print("No detections found")
    #     continue

    if frame_data and frame_count % fps == 0:
        sorted_frame_data = sorted(frame_data, key=lambda x: x[1])
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_frame_data)

    active_objects = new_active_objects
    total_perimeter = sum(distances)
    cv2.putText(frame, f"Perimeter: {total_perimeter:.1f} ft", (10, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y_offset = 30
    for class_name, count in class_counts.items():
        down_count = class_counts_direction['top_to_bottom'][class_name]
        up_count = class_counts_direction['bottom_to_top'][class_name]
        cv2.putText(frame, f"{class_name}: {count} (bottom:{down_count}, top:{up_count})",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow('Polygon on Frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
