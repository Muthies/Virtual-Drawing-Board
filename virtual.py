import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define drawing colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
current_color = (255, 255, 255)  # Default White

# Initialize variables
canvas = None
drawing = False
clear = False
prev_fingertip_pos = None  # To track the previous position of the fingertip
smoothed_pos = None  # For low-pass filtering

# Function to detect finger tip (Index Finger)
def get_finger_tip(hand_landmarks, w, h):
    index_tip = hand_landmarks.landmark[8]  # Index finger tip
    return int(index_tip.x * w), int(index_tip.y * h)

# Main Application
def main():
    global canvas, current_color, drawing, clear, prev_fingertip_pos, smoothed_pos

    cap = cv2.VideoCapture(0)
    print("Instructions:")
    print("- Point at color buttons to select color.")
    print("- Use index finger to draw.")
    print("- Point at 'Clear' button to erase the canvas.")
    print("- Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame and get dimensions
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Initialize the canvas if not done
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get fingertip position
                fingertip_x, fingertip_y = get_finger_tip(hand_landmarks, w, h)

                # Apply smoothing (low-pass filter)
                if smoothed_pos is None:
                    smoothed_pos = (fingertip_x, fingertip_y)
                else:
                    alpha = 0.2  # Smoothing factor
                    smoothed_pos = (
                        int(smoothed_pos[0] * (1 - alpha) + fingertip_x * alpha),
                        int(smoothed_pos[1] * (1 - alpha) + fingertip_y * alpha),
                    )

                # Color selection
                for i, color in enumerate(colors):
                    if 10 + i * 70 < fingertip_x < 60 + i * 70 and 10 < fingertip_y < 60:
                        current_color = color
                        drawing = False
                        prev_fingertip_pos = None
                        smoothed_pos = None
                        break

                # Clear button
                if w - 100 < fingertip_x < w - 10 and 10 < fingertip_y < 60:
                    clear = True
                    prev_fingertip_pos = None
                    smoothed_pos = None
                    break

                # Draw if index finger is raised
                drawing = True

                if drawing:
                    if prev_fingertip_pos is not None:
                        # Draw a line between the previous and smoothed position
                        cv2.line(canvas, prev_fingertip_pos, smoothed_pos, current_color, 2)

                    prev_fingertip_pos = smoothed_pos

        # Clear canvas if clear is activated
        if clear:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            clear = False

        # Add buttons for color selection
        for i, color in enumerate(colors):
            cv2.rectangle(frame, (10 + i * 70, 10), (60 + i * 70, 60), color, -1)

        # Add "Clear" button
        cv2.rectangle(frame, (w - 100, 10), (w - 10, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Clear", (w - 90, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Combine canvas and frame
        combined_frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

        # Display the result
        cv2.imshow("Virtual Drawing Board", combined_frame)

        # Exit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
