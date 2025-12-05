import speech_recognition as sr

def ouvir_microfone():
    reconhecedor = sr.Recognizer()
    print(sr.Microphone.list_microphone_names())

    with sr.Microphone() as source:
        print("Calibrando ruído ambiente... aguarde um segundo.")
        reconhecedor.adjust_for_ambient_noise(source)
        
        print("\n--- Pode falar agora (Ouvindo...) ---")
        
        try:
            audio = reconhecedor.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Processando áudio...")
            
            texto = reconhecedor.recognize_google(audio, language='pt-BR')
            
            print("\nVocê disse: " + texto)
            return texto

        except sr.WaitTimeoutError:
            print("Não ouvi nada dentro do tempo limite.")
        except sr.UnknownValueError:
            print("Não entendi o que você disse.")
        except sr.RequestError as e:
            print(f"Erro na conexão com o serviço de reconhecimento; {e}")

if __name__ == "__main__":
    while True:
        ouvir_microfone()
        continuar = input("\nDeseja tentar novamente? (s/n): ")
        if continuar.lower() != 's':
            break