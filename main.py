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



video_path = os.path.join('assets', 'input_video.mp4')

def calculate_distance_feet(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates in feet"""
    R = 20902231  # Earth's radius in feet
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def calculate_pixel_to_feet_ratio(roi_points, point_coordinates):
    """Calculate the pixel to feet ratio using ROI points and their GPS coordinates"""
    pixel_distances = []
    real_distances = []
    
    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]
        
        # Calculate pixel distance
        pixel_dist = sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pixel_distances.append(pixel_dist)
        
        # Calculate real distance using GPS
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        real_dist = calculate_distance_feet(lat1, lon1, lat2, lon2)
        real_distances.append(real_dist)
    
    # Calculate average ratio
    ratios = [real/pixel for real, pixel in zip(real_distances, pixel_distances)]
    return sum(ratios) / len(ratios)

def interpolate_gps(x, y, roi_points, point_coordinates):
    """Interpolate GPS coordinates for a point inside ROI"""
    # Calculate relative position within ROI
    total_area = 0
    weighted_lat = 0
    weighted_lon = 0
    
    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]
        pt3 = (x, y)
        
        # Calculate triangle area
        area = abs((pt1[0]*(pt2[1]-pt3[1]) + pt2[0]*(pt3[1]-pt1[1]) + pt3[0]*(pt1[1]-pt2[1]))/2)
        
        # Get GPS coordinates
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        
        # Weight coordinates by area
        avg_lat = (lat1 + lat2) / 2
        avg_lon = (lon1 + lon2) / 2
        
        weighted_lat += avg_lat * area
        weighted_lon += avg_lon * area
        total_area += area
    
    if total_area > 0:
        return weighted_lat/total_area, weighted_lon/total_area
    return None

def draw_distances(frame, roi_points, point_coordinates):
    """Draw distances on each edge of the ROI"""
    distances = []
    for i in range(len(roi_points)):
        pt1 = roi_points[i]
        pt2 = roi_points[(i + 1) % len(roi_points)]
        
        # Get GPS coordinates for the points
        lat1, lon1 = point_coordinates[pt1]
        lat2, lon2 = point_coordinates[pt2]
        
        # Calculate distance
        dist = calculate_distance_feet(lat1, lon1, lat2, lon2)
        distances.append(dist)
        
        # Calculate midpoint for text placement
        mid_x = (pt1[0] + pt2[0]) // 2
        mid_y = (pt1[1] + pt2[1]) // 2
        
        # Draw distance on frame
        # White text with black outline for better visibility
        text = f"{dist:.1f} ft"
        cv2.putText(frame, text, 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 3)  # Black outline
        cv2.putText(frame, text, 
                    (mid_x, mid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 1)  # White text
    
    return distances

def get_perspective_transform(source_points, target_points):
    """Get perspective transform matrix"""
    source = np.array(source_points, dtype=np.float32)
    target = np.array(target_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(source, target)

def transform_points(points, transform_matrix):
    """Transform points using perspective transform matrix"""
    if len(points) == 0:
        return points
    
    # Reshape points for perspective transform
    reshaped_points = np.array(points).reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, transform_matrix)
    return transformed_points.reshape(-1, 2)

def setup_perspective_transform(roi_points):
    """Setup perspective transform using ROI points"""
    # Get bounding rectangle of ROI
    x_min = min(pt[0] for pt in roi_points)
    x_max = max(pt[0] for pt in roi_points)
    y_min = min(pt[1] for pt in roi_points)
    y_max = max(pt[1] for pt in roi_points)
    
    width = x_max - x_min
    height = y_max - y_min
    
    # Define target points for bird's eye view
    target_points = np.array([
        [x_min, y_min],  # Top-left
        [x_min + width, y_min],  # Top-right
        [x_min + width, y_min + height],  # Bottom-right
        [x_min, y_min + height]  # Bottom-left
    ], dtype=np.float32)
    
    return get_perspective_transform(roi_points, target_points)

def calculate_speed(prev_pos, curr_pos, time_elapsed, pixel_to_feet, transform_matrix):
    """Calculate speed using transformed points"""
    if not time_elapsed:
        return 0
    
    # Transform points to bird's eye view
    transformed_points = transform_points([prev_pos, curr_pos], transform_matrix)
    prev_transformed = transformed_points[0]
    curr_transformed = transformed_points[1]
    
    # Calculate distance in transformed space
    dx = curr_transformed[0] - prev_transformed[0]
    dy = curr_transformed[1] - prev_transformed[1]
    pixel_dist = np.sqrt(dx*dx + dy*dy)
    
    # Convert to feet and calculate speed
    feet_dist = pixel_dist * pixel_to_feet
    speed_mph = (feet_dist / time_elapsed) * 0.681818
    return speed_mph

# Global variables for ROI and line selection
roi_points = []
line_points = []  # For storing line start and end points
is_selecting_roi = True
is_selecting_line = False
point_coordinates = {}

def click_event(event, x, y, flags, param):
    global roi_points, line_points, is_selecting_roi, is_selecting_line
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_selecting_roi:
            # Get lat/long input from user
            coords = input(f"Enter latitude,longitude for point ({x},{y}): ")
            try:
                lat, lon = map(float, coords.split(','))
                roi_points.append((x, y))
                point_coordinates[(x, y)] = (lat, lon)
                print(f"Added vehicle ROI point: ({x},{y}) with coordinates: {lat},{lon}")
                if len(roi_points) == 4:
                    is_selecting_roi = False
                    is_selecting_line = True
                    print("Now select 2 points for counting line")
            except ValueError:
                print("Invalid input! Please enter latitude,longitude in correct format")
                return
        elif is_selecting_line:
            line_points.append((x, y))
            print(f"Added line point: ({x},{y})")
            if len(line_points) == 2:
                is_selecting_line = False

# Load the video and get first frame
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# Create window and set mouse callback
cv2.namedWindow('ROI Selection')
cv2.setMouseCallback('ROI Selection', click_event)

print("Select 4 points for Vehicle ROI")
print("Enter coordinates in format: latitude,longitude")

while is_selecting_roi or is_selecting_line:
    temp_frame = frame.copy()
    
    # Draw ROI points and lines
    if len(roi_points) > 0:
        for point in roi_points:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)
            if point in point_coordinates:
                lat, lon = point_coordinates[point]
                cv2.putText(temp_frame, f"{lat:.6f}, {lon:.6f}", 
                           (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(roi_points) > 1:
            cv2.polylines(temp_frame, [np.array(roi_points, np.int32)], 
                         True, (0, 0, 255), 2)

    # Draw line points and line
    if len(line_points) > 0:
        for point in line_points:
            cv2.circle(temp_frame, point, 5, (255, 0, 255), -1)
        if len(line_points) == 2:
            cv2.line(temp_frame, line_points[0], line_points[1], (255, 0, 255), 2)

    # Display progress
    if is_selecting_roi:
        cv2.putText(temp_frame, f"Vehicle ROI Points: {len(roi_points)}/4", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif is_selecting_line:
        cv2.putText(temp_frame, f"Line Points: {len(line_points)}/2", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ROI Selection', temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow('ROI Selection')

# Create polygon from selected points
roi_polygon = Polygon(roi_points)

# Set line points for counting
line_start = line_points[0]
line_end = line_points[1]

# Save the coordinates to a file
with open('roi_coordinates.txt', 'w') as f:
    f.write("Vehicle ROI Points:\n")
    for point in roi_points:
        lat, lon = point_coordinates[point]
        f.write(f"Point {point}: {lat},{lon}\n")

# Get FPS and calculate smoothing window
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Use integer division (//) to handle odd FPS numbers
SMOOTHING_WINDOW = max(3, fps // 2)  # Half second window, minimum of 3 frames

# Calculate pixel to feet ratio
PIXEL_TO_FEET = calculate_pixel_to_feet_ratio(roi_points, point_coordinates)

# Setup perspective transform matrix
TRANSFORM_MATRIX = setup_perspective_transform(roi_points)

print(f"Video FPS: {fps}")
print(f"Calculated PIXEL_TO_FEET ratio: {PIXEL_TO_FEET:.4f}")
print(f"Using smoothing window of {SMOOTHING_WINDOW} frames")

# Path to the video
output_path = os.path.join(
    'assets', 'output_video_with_polygon.avi')  # Output file

# Load the model
model = YOLO('yolo11n.pt')
model.to(DEVICE)  # Move model to the selected device


# Get the class list
class_list = model.names

print(width, height, fps)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define counting line and region of interest (ROI)
line_color = (255, 0, 255)
line_thickness = 2

# Counter and trajectory storage
crossed_ids = set()
crossed_ids_down = set()  # For objects crossing from top to bottom
crossed_ids_up = set()    # For objects crossing from bottom to top
class_counts = defaultdict(int)
class_counts_direction = {
    'top_to_bottom': defaultdict(int),
    'bottom_to_top': defaultdict(int)
}
trajectories = defaultdict(lambda: deque())  # Stores full trajectory from entry to exit
active_objects = set()  # Tracks objects currently in ROI
object_speeds = {}  # Stores speed of objects
previous_positions = {}  # Stores previous positions of objects along with timestamps
speed_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))  # Stores speed history for smoothing

frame_count = 0  # Add frame_count variable

# Create CSV file and write headers
csv_filename = f'tracking_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Object_ID', 'Object_Type', 'Speed_MPH', 
                    'Current_Lat', 'Current_Long', 
                    'Trajectory_Points'])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    frame_count += 1  # Increment frame_count
    
    # Run YOLO tracking on the frame
    results = model.track(frame, persist=True, classes=[
                          0, 1, 2, 3, 5, 6, 7], device=DEVICE)

    # Draw the ROI polygons and counting lines
    cv2.polylines(frame, [np.array(roi_points, np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.line(frame, line_start, line_end, line_color, line_thickness)

    # Draw distances on edges
    distances = draw_distances(frame, roi_points, point_coordinates)
    
    # Print distances in console every 30 frames
    if frame_count % 30 == 0:
        print("\nEdge distances:")
        for i, dist in enumerate(distances):
            print(f"Edge {i+1}: {dist:.1f} ft")
        print(f"Total perimeter: {sum(distances):.1f} ft")

    new_active_objects = set()
    
    # Initialize list to store current frame data
    frame_data = []
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if results[0].boxes.data is not None:
        # Get the detected boxes, their class indices, and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        # Loop through each detected object 
        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2  # Calculate the center point
            cy = (y1 + y2) // 2

            class_name = class_list[class_idx]

            # Check if the object is inside the ROI
            point = Point(cx, cy)
            if not roi_polygon.contains(point):
                if track_id in trajectories:
                    del trajectories[track_id]  # Remove trajectory if object leaves ROI
                continue  # Skip tracking objects outside the ROI

            new_active_objects.add(track_id)
            trajectories[track_id].append((cx, cy))

            # Calculate speed using transformed points
            if track_id in previous_positions:
                prev_pos, prev_time = previous_positions[track_id]
                curr_pos = (cx, cy)
                time_elapsed = current_time - prev_time
                
                # Calculate speed using transformed points
                current_speed = calculate_speed(
                    prev_pos, 
                    curr_pos, 
                    time_elapsed, 
                    PIXEL_TO_FEET,
                    TRANSFORM_MATRIX
                )
                
                # Add to speed history for smoothing
                speed_history[track_id].append(current_speed)
                
                # Calculate smoothed speed
                if len(speed_history[track_id]) >= 3:
                    smoothed_speed = sum(speed_history[track_id]) / len(speed_history[track_id])
                    object_speeds[track_id] = round(smoothed_speed, 1)

            previous_positions[track_id] = ((cx, cy), current_time)

            # Update the label to show speed in mph
            label = f"ID: {track_id} | {class_name} | {object_speeds.get(track_id, 0):.1f} mph"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - 30), (x1 + text_size[0] + 10, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw full trajectory from entry to exit
            if track_id in trajectories:
                for i in range(1, len(trajectories[track_id])):
                    if trajectories[track_id][i - 1] is None or trajectories[track_id][i] is None:
                        continue
                    cv2.line(frame, trajectories[track_id][i - 1], trajectories[track_id][i], (255, 255, 0), 2)

            # Check if the object crosses the counting line
            if track_id not in crossed_ids_down and track_id not in crossed_ids_up:
                # Get previous position
                prev_pos = trajectories[track_id][-2] if len(trajectories[track_id]) > 1 else None
                
                if prev_pos:
                    prev_x, prev_y = prev_pos
                    curr_x, curr_y = cx, cy
                    
                    # Calculate line y-coordinate at current x position
                    # Using linear interpolation between line start and end points
                    line_slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
                    line_y = line_start[1] + line_slope * (cx - line_start[0])
                    
                    # Check if line is crossed
                    if (prev_y < line_y and curr_y >= line_y) or (prev_y > line_y and curr_y <= line_y):
                        # Determine direction of crossing
                        if prev_y < curr_y:  # Top to bottom
                            crossed_ids_down.add(track_id)
                            class_counts_direction['top_to_bottom'][class_name] += 1
                            direction = "Top to Bottom"
                        else:  # Bottom to top
                            crossed_ids_up.add(track_id)
                            class_counts_direction['bottom_to_top'][class_name] += 1
                            direction = "Bottom to Top"
                            
                        print(f"Counted: {class_name}, ID: {track_id}, Direction: {direction}")
                        class_counts[class_name] += 1

            # Get interpolated GPS coordinates for current position
            current_gps = interpolate_gps(cx, cy, roi_points, point_coordinates)
            
            if current_gps:
                # Get the latest trajectory point coordinates
                latest_point = trajectories[track_id][-1]  # Get last point (x,y)
                latest_gps = interpolate_gps(latest_point[0], latest_point[1], roi_points, point_coordinates)
                
                if latest_gps:
                    # Prepare data for CSV
                    frame_data.append([
                        current_timestamp,
                        track_id,
                        class_name,
                        object_speeds.get(track_id, 0),
                        latest_gps[0],  # latitude from latest trajectory point
                        latest_gps[1],  # longitude from latest trajectory point
                        f"({latest_gps[0]:.6f}, {latest_gps[1]:.6f})"  # latest point formatted as string
                    ])

    # Write frame data to CSV if there are any objects to report
    if frame_data and frame_count % fps == 0:  # Write once per second
        # Sort frame_data by object ID (track_id is at index 1)
        sorted_frame_data = sorted(frame_data, key=lambda x: x[1])
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(sorted_frame_data)

    active_objects = new_active_objects

    # Display total perimeter and area
    total_perimeter = sum(distances)
    cv2.putText(frame, f"Perimeter: {total_perimeter:.1f} ft", 
                (10, height - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 255), 2)

    # Display the counts on the frame
    y_offset = 30
    for class_name, count in class_counts.items():
        down_count = class_counts_direction['top_to_bottom'][class_name]
        up_count = class_counts_direction['bottom_to_top'][class_name]
        cv2.putText(frame, f"{class_name}: {count} (bottom:{down_count}, top:{up_count})", 
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    # Display the frame
    cv2.imshow('Polygon on Frame', frame)

    # Write the frame to the output video
    out.write(frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
