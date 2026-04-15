import json
import random
import threading
import time
import os
from datetime import datetime
from flask import Flask, request, Response

import vk_api
from vk_api.utils import get_random_id
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.exceptions import ApiError
from vk_api import VkUpload

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

# ================== НАСТРОЙКИ ==================
GROUP_ID = 237302992  # ID вашей группы
VK_TOKEN = os.getenv("VK_TOKEN")
CONFIRMATION_TOKEN = os.getenv("VK_CONFIRMATION_TOKEN")
DATA_FILE = "../data.json"

# ================== ИНИЦИАЛИЗАЦИЯ ==================
app = Flask(__name__)

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
upload = VkUpload(vk_session)

states = {}  # Состояния пользователей

# ================== РАБОТА С ДАННЫМИ ==================
def load_data():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = load_data()

def ensure_user(user_id):
    if str(user_id) not in data:
        data[str(user_id)] = {
            "deadlines": [],
            "notes": [],
            "grades": {}  # Оценки для интеграции с ЭПОС
        }

# ================== ОТПРАВКА СООБЩЕНИЙ ==================
def send_message(user_id, text, keyboard=None, attachment=None):
    try:
        vk.messages.send(
            user_id=user_id,
            message=text,
            random_id=get_random_id(),
            keyboard=keyboard,
            attachment=attachment
        )
    except ApiError as e:
        print("Ошибка отправки:", e)

# ================== КЛАВИАТУРЫ ==================
def main_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("➕ Дедлайн", VkKeyboardColor.POSITIVE)
    keyboard.add_button("📅 Мои дедлайны", VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("➕ Заметка", VkKeyboardColor.POSITIVE)
    keyboard.add_button("📝 Мои заметки", VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("🔍 AI-поиск", VkKeyboardColor.PRIMARY)
    keyboard.add_button("📊 Успеваемость", VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    keyboard.add_button("➕ Оценка", VkKeyboardColor.POSITIVE)
    keyboard.add_button("❌ Отмена", VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

def delete_deadline_keyboard(user_id):
    keyboard = VkKeyboard(inline=True)
    for i, dl in enumerate(data[str(user_id)]["deadlines"]):
        keyboard.add_callback_button(
            label=f"Удалить {i+1}",
            color=VkKeyboardColor.NEGATIVE,
            payload={"action": "delete_deadline", "index": i}
        )
        keyboard.add_line()
    return keyboard.get_keyboard()

def delete_note_keyboard(user_id):
    keyboard = VkKeyboard(inline=True)
    for i, note in enumerate(data[str(user_id)]["notes"]):
        keyboard.add_callback_button(
            label=f"Удалить {i+1}",
            color=VkKeyboardColor.NEGATIVE,
            payload={"action": "delete_note", "index": i}
        )
        keyboard.add_line()
    return keyboard.get_keyboard()

# ================== AI-ПОИСК ==================
def search_notes(notes, query):
    if not notes:
        return []

    texts = [note["text"] for note in notes]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts + [query])
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    results = []
    for i, score in enumerate(similarities):
        if score > 0.1:
            results.append(notes[i])
    return results

# ================== НАПОМИНАНИЯ ==================
def reminder_loop():
    while True:
        now = datetime.now()
        for user_id, user_data in data.items():
            for deadline in user_data["deadlines"]:
                if not deadline["notified"]:
                    dl_time = datetime.strptime(deadline["datetime"], "%d.%m.%Y %H:%M")
                    if now >= dl_time:
                        send_message(
                            int(user_id),
                            f"⏰ Напоминание!\n{deadline['text']}\n{deadline['datetime']}",
                            main_keyboard()
                        )
                        deadline["notified"] = True
                        save_data(data)
        time.sleep(60)

threading.Thread(target=reminder_loop, daemon=True).start()

# ================== ГРАФИК УСПЕВАЕМОСТИ ==================
def generate_performance_chart(user_id):
    grades = data[str(user_id)].get("grades", {})
    if not grades:
        return None

    plt.figure(figsize=(8, 5))
    for subject, marks in grades.items():
        plt.plot(range(1, len(marks) + 1), marks, marker='o', label=subject)

    plt.title("График успеваемости")
    plt.xlabel("Номер оценки")
    plt.ylabel("Оценка")
    plt.ylim(1, 5)
    plt.grid(True)
    plt.legend()

    filename = f"performance_{user_id}.png"
    plt.savefig(filename)
    plt.close()
    return filename

def upload_photo(file_path):
    photo = upload.photo_messages(file_path)[0]
    attachment = f"photo{photo['owner_id']}_{photo['id']}"
    os.remove(file_path)
    return attachment

# ================== CALLBACK API ==================
@app.route('/', methods=['POST'])
def callback():
    event = request.get_json(force=True)

    # Подтверждение сервера
    if event['type'] == 'confirmation':
        return Response(CONFIRMATION_TOKEN, status=200, mimetype='text/plain')

    # Обработка inline-кнопок
    if event['type'] == 'message_event':
        payload = event['object']['payload']
        user_id = event['object']['user_id']
        ensure_user(user_id)

        if payload['action'] == 'delete_deadline':
            index = payload['index']
            try:
                removed = data[str(user_id)]["deadlines"].pop(index)
                save_data(data)
                send_message(user_id, f"✅ Дедлайн удалён: {removed['text']}", main_keyboard())
            except IndexError:
                send_message(user_id, "❌ Ошибка удаления.", main_keyboard())

        elif payload['action'] == 'delete_note':
            index = payload['index']
            try:
                data[str(user_id)]["notes"].pop(index)
                save_data(data)
                send_message(user_id, "✅ Заметка удалена.", main_keyboard())
            except IndexError:
                send_message(user_id, "❌ Ошибка удаления.", main_keyboard())

        vk.messages.sendMessageEventAnswer(
            event_id=event['object']['event_id'],
            user_id=user_id,
            peer_id=event['object']['peer_id']
        )
        return Response('ok', status=200)

    # Обработка сообщений
    if event['type'] == 'message_new':
        message = event['object']['message']
        user_id = message['from_id']
        text = message['text'].lower().strip()
        attachments = message.get('attachments', [])

        ensure_user(user_id)

        # Отмена действия
        if text in ["отмена", "❌ отмена"]:
            states[user_id] = None
            send_message(user_id, "❌ Действие отменено.", main_keyboard())
            return Response('ok', status=200)

        # Приветствие
        if text in ["привет", "начать", "start"]:
            send_message(
                user_id,
                "👋 Привет! Я бот для управления дедлайнами, заметками и отслеживания успеваемости.",
                main_keyboard()
            )
            return Response('ok', status=200)

        # Добавление дедлайна
        if text == "➕ дедлайн":
            states[user_id] = "add_deadline"
            send_message(
                user_id,
                "Введите дедлайн в формате:\nДД.ММ.ГГГГ ЧЧ:ММ Описание",
                main_keyboard()
            )
            return Response('ok', status=200)

        # Просмотр дедлайнов
        if text == "📅 мои дедлайны":
            deadlines = data[str(user_id)]["deadlines"]
            if not deadlines:
                send_message(user_id, "У вас нет дедлайнов.", main_keyboard())
            else:
                msg = "📅 Ваши дедлайны:\n\n"
                for i, dl in enumerate(deadlines, 1):
                    msg += f"{i}. {dl['datetime']} — {dl['text']}\n"
                send_message(user_id, msg, delete_deadline_keyboard(user_id))
            return Response('ok', status=200)

        # Добавление заметки
        if text == "➕ заметка":
            states[user_id] = "add_note"
            send_message(user_id, "Отправьте фотографию конспекта с описанием.", main_keyboard())
            return Response('ok', status=200)

        # Просмотр заметок
        if text == "📝 мои заметки":
            notes = data[str(user_id)]["notes"]
            if not notes:
                send_message(user_id, "У вас нет заметок.", main_keyboard())
            else:
                for i, note in enumerate(notes, 1):
                    send_message(
                        user_id,
                        f"{i}. {note['text']}",
                        delete_note_keyboard(user_id),
                        attachment=note['photo']
                    )
            return Response('ok', status=200)

        # AI-поиск
        if text == "🔍 ai-поиск":
            states[user_id] = "search"
            send_message(user_id, "Введите текст для поиска.", main_keyboard())
            return Response('ok', status=200)

        # Добавление оценки (ЭПОС)
        if text == "➕ оценка":
            states[user_id] = "add_grade"
            send_message(
                user_id,
                "Введите оценку в формате:\nПредмет Оценка\nПример: Математика 5",
                main_keyboard()
            )
            return Response('ok', status=200)

        # График успеваемости
        if text == "📊 успеваемость":
            chart_path = generate_performance_chart(user_id)
            if chart_path:
                attachment = upload_photo(chart_path)
                send_message(user_id, "📊 Ваш график успеваемости:", main_keyboard(), attachment)
            else:
                send_message(user_id, "❌ У вас пока нет добавленных оценок.", main_keyboard())
            return Response('ok', status=200)

        # Обработка состояний
        state = states.get(user_id)

        if state == "add_deadline":
            try:
                parts = message['text'].split(" ", 2)
                dt = parts[0] + " " + parts[1]
                desc = parts[2]
                datetime.strptime(dt, "%d.%m.%Y %H:%M")

                data[str(user_id)]["deadlines"].append({
                    "datetime": dt,
                    "text": desc,
                    "notified": False
                })
                save_data(data)
                states[user_id] = None
                send_message(user_id, "✅ Дедлайн добавлен!", main_keyboard())
            except Exception:
                send_message(user_id, "❌ Неверный формат даты.", main_keyboard())
            return Response('ok', status=200)

        elif state == "add_note":
            if attachments and attachments[0]["type"] == "photo":
                photo = attachments[0]["photo"]
                photo_id = f"photo{photo['owner_id']}_{photo['id']}"

                data[str(user_id)]["notes"].append({
                    "text": message['text'],
                    "photo": photo_id
                })
                save_data(data)
                states[user_id] = None
                send_message(user_id, "✅ Заметка сохранена!", main_keyboard())
            else:
                send_message(user_id, "❌ Отправьте фотографию.", main_keyboard())
            return Response('ok', status=200)

        elif state == "search":
            results = search_notes(data[str(user_id)]["notes"], message['text'])
            if results:
                for note in results:
                    send_message(
                        user_id,
                        f"🔍 Найдено: {note['text']}",
                        attachment=note['photo']
                    )
            else:
                send_message(user_id, "❌ Ничего не найдено.", main_keyboard())
            states[user_id] = None
            return Response('ok', status=200)

        elif state == "add_grade":
            try:
                subject, grade = message['text'].rsplit(" ", 1)
                grade = int(grade)

                if subject not in data[str(user_id)]["grades"]:
                    data[str(user_id)]["grades"][subject] = []

                data[str(user_id)]["grades"][subject].append(grade)
                save_data(data)
                states[user_id] = None
                send_message(user_id, "✅ Оценка успешно добавлена!", main_keyboard())
            except Exception:
                send_message(user_id, "❌ Неверный формат. Попробуйте снова.", main_keyboard())
            return Response('ok', status=200)

        send_message(user_id, "Выберите действие:", main_keyboard())
        return Response('ok', status=200)

    return Response('ok', status=200)

# ================== ЗАПУСК ==================
def main():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()