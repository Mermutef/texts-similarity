import requests


def main():
    server_url = "http://localhost:5000/compare"

    student_answer = input("Введите ответ студента: ")

    reference_answer = input("Введите эталонный ответ: ")

    data = {
        "student_answer": student_answer,
        "reference_answer": reference_answer
    }

    try:
        response = requests.post(server_url, json=data)

        if response.status_code == 200:
            result = response.json()
            print(f"Семантическая схожесть: {result['similarity']:.4f}")
        else:
            print(f"Ошибка сервера: {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Ошибка соединения с сервером: {e}")


if __name__ == "__main__":
    while True:
        main()
        if input("Продолжить? [д]/н: ") == "н":
            break
