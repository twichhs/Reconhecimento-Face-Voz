import cv2
import face_recognition
import os
import numpy as np

PASTA_ROSTOS = "rostos_conhecidos"
ESCALA_PROCESSAMENTO = 0.50

def carregar_rostos_conhecidos():
    """Lê a pasta e aprende os rostos que estão lá."""
    codificacoes_conhecidas = []
    nomes_conhecidos = []

    if not os.path.exists(PASTA_ROSTOS):
        os.makedirs(PASTA_ROSTOS)
        print(f"AVISO: Pasta '{PASTA_ROSTOS}' criada. Coloque fotos lá dentro!")
        return [], []

    print("Carregando e aprendendo rostos conhecidos...")
    
    for arquivo in os.listdir(PASTA_ROSTOS):
        if arquivo.endswith(('.jpg', '.jpeg', '.png')):
            caminho_imagem = os.path.join(PASTA_ROSTOS, arquivo)
            
            try:
                imagem_rosto = face_recognition.load_image_file(caminho_imagem)
                
                codificacao = face_recognition.face_encodings(imagem_rosto)[0]
                
                codificacoes_conhecidas.append(codificacao)
                nome = os.path.splitext(arquivo)[0]
                nomes_conhecidos.append(nome.replace("_", " ").title())
                print(f" -> Aprendi o rosto de: {nome}")
            except IndexError:
                print(f"ERRO: Não consegui achar um rosto na imagem {arquivo}. Verifique a foto.")
            except Exception as e:
                print(f"ERRO ao processar {arquivo}: {e}")
                
    print(f"Concluído. {len(nomes_conhecidos)} rostos aprendidos.\n")
    return codificacoes_conhecidas, nomes_conhecidos


def iniciar_reconhecimento():
    base_codificacoes, base_nomes = carregar_rostos_conhecidos()

    if not base_codificacoes:
        print("Nenhum rosto conhecido para identificar. Adicione fotos na pasta.")

    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Erro ao abrir webcam.")
        return

    print("Reconhecimento Facial PRO iniciado. Pressione 'q' para sair.")

    while True:
        ret, frame = video_capture.read()
        if not ret: break

        small_frame = cv2.resize(frame, (0, 0), fx=ESCALA_PROCESSAMENTO, fy=ESCALA_PROCESSAMENTO)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(base_codificacoes, face_encoding, tolerance=0.6)
            name = "Desconhecido"

            face_distances = face_recognition.face_distance(base_codificacoes, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = base_nomes[best_match_index]

            face_names.append(name)

        fator_escala = int(1/ESCALA_PROCESSAMENTO)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= fator_escala
            right *= fator_escala
            bottom *= fator_escala
            left *= fator_escala

            color = (0, 255, 0) if name != "Desconhecido" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        cv2.imshow('Reconhecimento Facial Potente', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    iniciar_reconhecimento()