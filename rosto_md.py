import cv2
import mediapipe as mp

def iniciar_deteccao_potente():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro ao abrir a webcam.")
            return

        print("Detecção de Alta Precisão iniciada. Pressione 'q' para sair.")

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignorando frame vazio.")
                continue

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = face_detection.process(image_rgb)

            image.flags.writeable = True
            
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                    
            cv2.imshow('MediaPipe Face Detection (Potente)', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_deteccao_potente()