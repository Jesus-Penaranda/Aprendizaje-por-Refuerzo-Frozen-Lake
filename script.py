import sys
import os
import subprocess  

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    if not os.path.exists(script_path):
        print(f"Error: No se ha encontrado el archivo {script_name}")
        return
    print(f"\n{'='*50}")
    print(f"Lanzando: {script_name}")
    print(f"{'='*50}\n")

    subprocess.run([sys.executable, script_path])
    print(f"\n{'='*50}")
    print(f"Finalizado: {script_name}")
    print(f"{'='*50}\n")

def main():
    while True:
        print("1-Ejecutar Value Iteration (script_value_iteration.py)")
        print("2-Ejecutar Model Based (script_model_based.py)")
        print("3-Ejecutar Q-Learning (script_q_learning.py)") 
        print("4-Salir")
        
        opcion = input("Selecciona una opción (1-4): ").strip()
        
        if opcion == "1":
            run_script("script_value_iteration.py")
        elif opcion == "2":
            run_script("script_model_based.py")
        elif opcion == "3":
            run_script("script_q_learning.py") 
        elif opcion == "4":
            print("Saliendo...")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    main()