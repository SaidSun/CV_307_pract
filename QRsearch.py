import cv2
import numpy as np

def correct_perspective(img, src_points, dst_points, output_size):
    src = src_points
    dst = dst_points
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_image = cv2.warpPerspective(img, matrix, output_size)
    return warped_image

def calculate_tilt_angle(bbox):
    pts = bbox[0]
    
    pt1 = pts[2]
    pt2 = pts[3]
    
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    
    angle = np.degrees(np.arctan2(dy, dx))
    
    if angle < 0:
        angle += 180
    
    return angle

def decode_qr_code(frame):
    detector = cv2.QRCodeDetector()

    data, bbox, straight_qrcode = detector.detectAndDecode(frame)
    if bbox is not None:
        print(f"BBOX coords are {bbox}")
        base_coords = np.array([bbox[0][i] for i in range(len(bbox[0]))])
        width, height = 450, 350
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        warped_image = correct_perspective(frame, base_coords, dst_pts, (width, height))
        data_warped, bbox_warped, straight_qrcode_warped = detector.detectAndDecode(warped_image)
        if data:
            print(f"Decoded Data: {data_warped}")
            if bbox is not None:
                for i in range(len(bbox[0])):
                    pt1 = tuple(bbox[0][i].astype(int))
                    pt2 = tuple(bbox[0][(i+1) % len(bbox[0])].astype(int))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 3)
            pts = bbox[0]
            min_x = int(np.min(pts[:, 0]))
            min_y = int(np.min(pts[:, 1]))
            max_x = int(np.max(pts[:, 0]))
            max_y = int(np.max(pts[:, 1]))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            
            text_size = cv2.getTextSize(data, font, font_scale, thickness)[0]
            
            bg_x1, bg_y1 = min_x, min_y - text_size[1] - 10
            bg_x2, bg_y2 = min_x + text_size[0], min_y - 10
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
            
            cv2.putText(frame, data, (min_x, min_y - 10), font, font_scale, (0, 0, 0), thickness)

        else:
            print("QR Code not detected or could not be decoded.")
    return True

def main():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
            
        decode_qr_code(frame) 
        
        cv2.imshow('QR Code Scanner', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

