import telebot
from telebot import types
import sqlite3
from pymorphy2 import MorphAnalyzer
import re
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import nltk
from nltk.corpus import stopwords
from spellchecker import SpellChecker


def read_file(file_name):
    with open(file_name, 'r') as file:
        return file.read()


# bot = telebot.TeleBot('7056992273:AAHiATYByA-ItpAbwTMQo1kZk3tqJCb71Kc')
bot = telebot.TeleBot(read_file('token.ini'))
conn = sqlite3.connect('questions.db')
cur = conn.cursor()
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
cur_keys = [0, 0, 0, 0, 0, 0]


# подключаемся к бд
def connect_to_database():
    conn = sqlite3.connect('questions.db')
    return conn


# вытаскиваем рандомные вопросы из базы
def get_questions(conn, subj):
    cur = conn.cursor()
    if subj == 'charms':
        cur.execute("SELECT q, opA, opB, opC, key FROM charms1 ORDER BY RANDOM() LIMIT 4")
        questions = cur.fetchall()
        cur.execute("SELECT q, key, text FROM charms2 ORDER BY RANDOM() LIMIT 1")
        question = cur.fetchone()
    elif subj == 'potions':
        cur.execute("SELECT q, opA, opB, opC, key FROM potions1 ORDER BY RANDOM() LIMIT 4")
        questions = cur.fetchall()
        cur.execute("SELECT q, key, text FROM potions2 ORDER BY RANDOM() LIMIT 1")
        question = cur.fetchone()
    return questions, question


# ф-я, которая вытаскивает статистику прохождения из файла, чтобы построить график
# почему не база данных? если кратко, то многопоточность: никак не могу с ней разобраться, увы
def get_results():
    with open("results.txt", "r") as file:
        results = [int(line.strip()) for line in file]
    return results


# строим график по статистике
def plot_bar_chart(results):
    counts = [results.count(i) for i in range(6)]

    x = np.arange(6)
    fig, ax = plt.subplots()
    bars = ax.bar(x, counts, color='none', edgecolor='black')

    hatching = 'xx'

    for bar in bars:
        bar.set_hatch(hatching)

    plt.xlabel('Оценка', fontweight='bold')
    plt.ylabel('Количество человек', fontweight='bold')
    plt.title('Статистика прохождения теста', fontweight='bold')
    plt.xticks(x, [str(i) for i in range(6)])

    plt.gca().set_facecolor('#FFFAF0')

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer


# начало: тут наши главные основные команды
@bot.message_handler(func=lambda message: message.text in ['/start', '/help', '/stats'])
def handler(message):
    if message.text == '/start':
        hello = "Привет, юный волшебник! Здесь ты можешь проверить свое знание волшебных предметов Хогвартса и сдать" \
                " настоящие волшебные экзамены. Просто выбери предмет из списка ниже, ответь 5 вопросов и получи " \
                "оценку. \nЧтобы лучше узнать, как работает бот, напишите команду /help."
        keyboard1 = types.InlineKeyboardMarkup()  # создание клавиатуры
        key_charms = types.InlineKeyboardButton(text='Чары', callback_data='charms')
        keyboard1.add(key_charms)  # создание и добавление кнопки в клавиатуру
        key_potions = types.InlineKeyboardButton(text='Зелья', callback_data='potions')
        keyboard1.add(key_potions)
        bot.send_message(message.from_user.id, text=hello, reply_markup=keyboard1)
    if message.text == '/help':
        help_mes = "Привет! Это бот, в котором ты можешь пройти экзамены по предметам из Гарри Поттера. Сейчас " \
                   "расскажу, *как он работает*. " \
                   "\n\nРядом со скрепочкой есть кнопка Menu: там ты увидишь команды, которые есть у этого бота. " \
                   "*Вот что они делают*: " \
                   "\n\n•	/start – команда, при нажатии которой тебе предложат выбрать предмет, " \
                   "по которому ты хочешь проверить свои знания" \
                   "\n\n•	/help – это то, где ты сейчас находишься" \
                   "\n\n•	/stats – это общая статистика всех прохождений: при нажатии на эту команду тебе вылезет " \
                   "столбчатая диаграмма, на которой можно увидеть, как люди в целом справляются с тестами" \
                   "\n\n*Когда ты выбрал(а) предмет*" \
                   "\nСначала тебе выпадет инструкция. Если кратко, то у тебя будет 5 вопросов, " \
                   "на которые надо постараться ответить, а потом тебе сразу будет выдан результат. " \
                   "Потом тебе выпадут сами вопросы: отвечай на них, нажимая на прилагающиеся к ним команды, " \
                   "а по завершении получишь результат и веселый стикер. Потом ты можешь пройти тест еще раз. " \
                   "Вопросы рандомно выбираются из банка, поэтому набор будет каждый раз чуть-чуть разный. " \
                   "Тест не показывает, где именно у тебя ошибки. Удачи!"
        bot.send_message(message.chat.id, help_mes)
    if message.text == '/stats':
        results = get_results()
        chart_buffer = plot_bar_chart(results)
        bot.send_photo(message.chat.id, chart_buffer)


# перешли по кнопке в квиз по чарам или зельям
@bot.callback_query_handler(func=lambda call: call.data == "charms" or call.data == "potions")
def callback_worker(call):
    if call.data == "charms":  # квиз по чарам

        # главная ф-я запуска (запускаем внизу)
        def test(conn):
            keyboard_begin = types.InlineKeyboardMarkup()
            key_begin = types.InlineKeyboardButton(text="Начать тест!", callback_data='begin')
            keyboard_begin.add(key_begin)
            bot.send_message(call.from_user.id,
                             text='После начала теста тебе будет выведено 4 простых вопроса (с выбором ответа) '
                                  'и 1 сложный (с развернутым ответом). Перед каждым вопросом написана соответствующая '
                                  'команда. Нажми на команду и в следующем сообщении тебя попросят ввести ответ на '
                                  'соответствующий вопрос. Пожалуйста, не путай русскую и английскую раскладку и '
                                  'пиши без ошибок! Если захочешь поменять свой ответ, просто напиши '
                                  'соответствующую команду еще раз.'
                                  '\n\nВопросы с выбором ответа отмечены латинскими буквами: A, B и C.'
                                  ' Для ответа на эти вопросы, пожалуйста, пиши *только* букву, соответствующую '
                                  'правильному варианту ответа. Например, "A".'
                                  '\n\nПоследний вопрос требует короткого ответа своими словами. Также тебе будет '
                                  'выведено, насколько твой ответ похож на предполагаемый (что не является гарантией '
                                  'правильности или неправильности ответа). Основываясь на этом, ты сможешь поменять '
                                  'свою формулировку, если захочешь.'
                                  '\n\nЧтобы закончить и получить результат, напиши /finish.',
                             reply_markup=keyboard_begin, parse_mode="Markdown")

        # перешли по кнопке начать
        @bot.callback_query_handler(func=lambda call: call.data == "begin")
        def callback_worker2(call):
            if call.data == "begin":
                conn = connect_to_database()
                questions, question = get_questions(conn, 'charms')
                conn.close()

                u_answers = [0, 0, 0, 0, 0]
                correct = [questions[0][4], questions[1][4], questions[2][4], questions[3][4], question[1]]
                grade = [0, 0, 0, 0, 0]

                # выводим вопросы
                frage = "/1 " + questions[0][0] + "\n" + " A " + questions[0][1] + "\n" + " B " + questions[0][
                    2] + "\n" + \
                        " C " + questions[0][3] + "\n" + \
                        "/2 " + questions[1][0] + "\n" + " A " + questions[1][1] + "\n" + " B " + questions[1][
                            2] + "\n" + \
                        " C " + questions[1][3] + "\n" + \
                        "/3 " + questions[2][0] + "\n" + " A " + questions[2][1] + "\n" + " B " + questions[2][
                            2] + "\n" + \
                        " C " + questions[2][3] + "\n" + \
                        "/4 " + questions[3][0] + "\n" + " A " + questions[3][1] + "\n" + " B " + questions[3][
                            2] + "\n" + \
                        " C " + questions[3][3] + "\n" + \
                        "/5 " + question[0] + \
                        "\n" + "/finish - чтобы закончить"

                bot.send_message(call.message.chat.id, frage)

                # сохраняем в глобальном массиве ответы
                # иначе при повторном запуске все сломается,
                # и ответы на новые вопросы не перезапишутся
                cur_keys[0] = questions[0][4]
                cur_keys[1] = questions[1][4]
                cur_keys[2] = questions[2][4]
                cur_keys[3] = questions[3][4]
                cur_keys[4] = question[1]
                cur_keys[5] = question[2]

                # ф-я добавления результата в файл
                def add_result(result):
                    with open("results.txt", "a") as file:
                        file.write(str(result) + "\n")

                # ф-я проверки ответов
                def check_test(conn):

                    # это потому что при повторном запуске иначе все ломается (остаются старые данные)
                    for i in range(len(grade)):
                        grade[i] = 0

                    correct[0] = cur_keys[0]
                    correct[1] = cur_keys[1]
                    correct[2] = cur_keys[2]
                    correct[3] = cur_keys[3]
                    correct[4] = cur_keys[4]

                    # проверка тестовой части
                    for i in range(4):
                        if u_answers[i] == correct[i]:
                            grade[i] += 1

                    # проверка последнего вопроса:
                    # вычистили, лемматизировали, вытащили начальную форму
                    morph = MorphAnalyzer()
                    cleaned_text = re.sub(r'[^\w\s]', ' ', str(u_answers[4]))
                    q5 = set(correct[4].split(', '))
                    q_u5 = cleaned_text.split()
                    for i in range(len(q_u5)):
                        ana = morph.parse(q_u5[i])
                        first = ana[0]
                        q_u5[i] = first.normal_form
                    q_u5 = set(q_u5)
                    # убедились, что все ключевые слова есть в данном ответе (только тогда засчитываем)
                    if q5 <= q_u5:
                        grade[4] += 1

                    # добавляем результат в файлик
                    add_result(sum(grade))

                    # выводим результат
                    if sum(grade) <= 1:
                        bot.send_message(call.message.chat.id,
                                         f'Кажется, кто-то немного подзабыл всю магию: '
                                         f'твой результат {sum(grade)} баллов из 5. Но ничего страшного,'
                                         f'попробуй еще раз, и все точно получится!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy4tmAnE0aZQ53RDJUwkBWEDODcvGqgACOEIAAlXMGUinLk97qBbd6DQE")
                    if 2 <= sum(grade) <= 3:
                        bot.send_message(call.message.chat.id,
                                         f'Ого! Неплохо. Скоро ты будешь первоклассным магом! '
                                         f'Твой результат {sum(grade)} баллов из 5. В следующий '
                                         f'раз точно получится набрать 100%!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy49mAnFCDaQXhVfgXSKyeaKjVZ2hLQACTkkAAlNEEEjTpulOEhm9qjQE")
                    if sum(grade) >= 4:
                        bot.send_message(call.message.chat.id,
                                         f'Вот это да! Ты случайно не будущий профессор Дамблдор? '
                                         f'Твой результат {sum(grade)} баллов из 5. Поздравляю!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy41mAnE_9XVXy7GBxJwP8Of2MgH3QAACI0EAAkFbGUiWAZVBHkcZmjQE")
                    # обнуляем старые ответы - иначе все сломается
                    for i in range(len(u_answers)):
                        u_answers[i] = 0

                # ф-я и хендлер для записи ответа на вопрос 1
                def one(message):
                    u_answers[0] = message.text

                @bot.message_handler(func=lambda message: message.text == '/1')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на первый вопрос:')
                    bot.register_next_step_handler(mesg, one)

                # ф-я и хендлер для записи ответа на вопрос 2
                def two(message):
                    u_answers[1] = message.text

                @bot.message_handler(func=lambda message: message.text == '/2')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на второй вопрос:')
                    bot.register_next_step_handler(mesg, two)

                # ф-я и хендлер для записи ответа на вопрос 3
                def three(message):
                    u_answers[2] = message.text

                @bot.message_handler(func=lambda message: message.text == '/3')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на третий вопрос:')
                    bot.register_next_step_handler(mesg, three)

                # ф-я и хендлер для записи ответа на вопрос 4
                def four(message):
                    u_answers[3] = message.text

                @bot.message_handler(func=lambda message: message.text == '/4')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на четвертый вопрос:')
                    bot.register_next_step_handler(mesg, four)

                # это все ф-и и хендлер для записи ответа на вопрос 5 и его обработки
                # чистим текст, удаляем стоп-слова
                def preprocess_text(text):
                    tokens = text.lower().split()
                    tokens = [token.strip('.,?!«»"\'') for token in tokens if token not in stop_words]
                    morph = MorphAnalyzer()
                    for i in range(len(tokens)):
                        ana = morph.parse(tokens[i])
                        first = ana[0]
                        tokens[i] = first.normal_form
                    return tokens

                # считаем косинусное расстояние (извините, pycharm не дал мне загрузить НИ ОДНУ семантическую модель,
                # так что выкручиваемся как можем)
                def cosine_similarity(vector1, vector2):
                    dot_product = np.dot(vector1, vector2)
                    norm_vector1 = np.linalg.norm(vector1)
                    norm_vector2 = np.linalg.norm(vector2)
                    similarity = dot_product / (norm_vector1 * norm_vector2)
                    return similarity

                # проверка опечаток (эта штука работает не очень, но это лучшее, что я нашла для русского языка)
                def check_spelling(text):
                    spell = SpellChecker(language='ru')
                    misspelled = spell.unknown(text)
                    return misspelled

                def five(message):
                    u_answers[4] = message.text  # ответ пользователя
                    cor_ans = cur_keys[5]  # идеальный ответ

                    # их предобработка
                    tokens1 = preprocess_text(message.text)
                    tokens2 = preprocess_text(cor_ans)

                    # проверка ошибок
                    mistakes = check_spelling(tokens1)

                    # считаем косинусную близость
                    word_set = set(tokens1).union(tokens2)
                    vector1 = np.array([tokens1.count(word) for word in word_set])
                    vector2 = np.array([tokens2.count(word) for word in word_set])
                    similarity_score = cosine_similarity(vector1, vector2)

                    # формируем сообщение в зависимости от косинусной близости
                    if similarity_score >= 0:
                        per = round(similarity_score * 100, 2)
                        mes = "Твой ответ похож на идеальный на " + str(
                            per) + "%. Это не гарантия (не)правильности, но возможно ты захочешь его поменять."
                    else:
                        per = - round(similarity_score * 100, 2)
                        mes = "Твой ответ отличается от идеального на " + str(
                            per) + "%. Это не гарантия (не)правильности, но возможно ты захочешь его поменять."
                    # дополняем то же сообщение, если заподозрили очепятку
                    if len(mistakes) > 0:
                        mes += "\nДружеское предупреждение: есть некоторая вероятность, что в твоем тексте есть " \
                               "опечатки. Возможно, стоит перепроверить ответ?"
                    bot.send_message(call.message.chat.id, mes)  # выводим

                @bot.message_handler(func=lambda message: message.text == '/5')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на пятый вопрос:')
                    bot.register_next_step_handler(mesg, five)

                # если пользователь закончил, направляем проверять результаты
                @bot.message_handler(func=lambda message: message.text == '/finish')
                def redirect(message):
                    check_test(conn)

        test(conn)  # запускаем всю эту волынку

    if call.data == "potions":  # квиз по зельям
        # здесь все абсолютно то же самое, что и сверху, поэтому комментарии на этом прерываю

        def test(conn):
            keyboard_begin = types.InlineKeyboardMarkup()
            key_begin = types.InlineKeyboardButton(text="Начать тест!", callback_data='begin1')
            keyboard_begin.add(key_begin)
            bot.send_message(call.from_user.id,
                             text='После начала теста тебе будет выведено 4 простых вопроса (с выбором ответа) '
                                  'и 1 сложный (с развернутым ответом). Перед каждым вопросом написана соответствующая '
                                  'команда. Нажми на команду и в следующем сообщении тебя попросят ввести ответ на '
                                  'соответствующий вопрос. Пожалуйста, не путай русскую и английскую раскладку и '
                                  'пиши без ошибок! Если захочешь поменять свой ответ, просто напиши '
                                  'соответствующую команду еще раз.'
                                  '\n\nВопросы с выбором ответа отмечены латинскими буквами: A, B и C.'
                                  ' Для ответа на эти вопросы, пожалуйста, пиши *только* букву, соответствующую '
                                  'правильному варианту ответа. Например, "A".'
                                  '\n\nПоследний вопрос требует короткого ответа своими словами. Также тебе будет '
                                  'выведено, насколько твой ответ похож на предполагаемый (что не является гарантией '
                                  'правильности или неправильности ответа). Основываясь на этом, ты сможешь поменять '
                                  'свою формулировку, если захочешь.'
                                  '\n\nЧтобы закончить и получить результат, напиши /finish.',
                             reply_markup=keyboard_begin, parse_mode="Markdown")

        @bot.callback_query_handler(func=lambda call: call.data == "begin1")
        def callback_worker2(call):
            if call.data == "begin1":
                conn = connect_to_database()
                questions, question = get_questions(conn, 'potions')
                conn.close()

                u_answers = [0, 0, 0, 0, 0]
                correct = [questions[0][4], questions[1][4], questions[2][4], questions[3][4], question[1]]
                grade = [0, 0, 0, 0, 0]

                frage = "/1 " + questions[0][0] + "\n" + " A " + questions[0][1] + "\n" + " B " + questions[0][
                    2] + "\n" + \
                        " C " + questions[0][3] + "\n" + \
                        "/2 " + questions[1][0] + "\n" + " A " + questions[1][1] + "\n" + " B " + questions[1][
                            2] + "\n" + \
                        " C " + questions[1][3] + "\n" + \
                        "/3 " + questions[2][0] + "\n" + " A " + questions[2][1] + "\n" + " B " + questions[2][
                            2] + "\n" + \
                        " C " + questions[2][3] + "\n" + \
                        "/4 " + questions[3][0] + "\n" + " A " + questions[3][1] + "\n" + " B " + questions[3][
                            2] + "\n" + \
                        " C " + questions[3][3] + "\n" + \
                        "/5 " + question[0] + \
                        "\n" + "/finish - чтобы закончить"

                bot.send_message(call.message.chat.id, frage)
                cur_keys[0] = questions[0][4]
                cur_keys[1] = questions[1][4]
                cur_keys[2] = questions[2][4]
                cur_keys[3] = questions[3][4]
                cur_keys[4] = question[1]
                cur_keys[5] = question[2]

                def add_result(result):
                    with open("results.txt", "a") as file:
                        file.write(str(result) + "\n")

                def check_test(conn):
                    for i in range(len(grade)):
                        grade[i] = 0

                    correct[0] = cur_keys[0]
                    correct[1] = cur_keys[1]
                    correct[2] = cur_keys[2]
                    correct[3] = cur_keys[3]
                    correct[4] = cur_keys[4]

                    for i in range(4):
                        if u_answers[i] == correct[i]:
                            grade[i] += 1

                    morph = MorphAnalyzer()
                    cleaned_text = re.sub(r'[^\w\s]', ' ', str(u_answers[4]))
                    q5 = set(correct[4].split(', '))
                    q_u5 = cleaned_text.split()
                    for i in range(len(q_u5)):
                        ana = morph.parse(q_u5[i])
                        first = ana[0]
                        q_u5[i] = first.normal_form
                    q_u5 = set(q_u5)
                    if q5 <= q_u5:
                        grade[4] += 1

                    add_result(sum(grade))

                    if sum(grade) <= 1:
                        bot.send_message(call.message.chat.id,
                                         f'Кажется, кто-то немного подзабыл всю магию: '
                                         f'твой результат {sum(grade)} баллов из 5. Но ничего страшного,'
                                         f'попробуй еще раз, и все точно получится!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy4tmAnE0aZQ53RDJUwkBWEDODcvGqgACOEIAAlXMGUinLk97qBbd6DQE")
                    if 2 <= sum(grade) <= 3:
                        bot.send_message(call.message.chat.id,
                                         f'Ого! Неплохо. Скоро ты будешь первоклассным магом! '
                                         f'Твой результат {sum(grade)} баллов из 5. В следующий '
                                         f'раз точно получится набрать 100%!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy49mAnFCDaQXhVfgXSKyeaKjVZ2hLQACTkkAAlNEEEjTpulOEhm9qjQE")
                    if sum(grade) >= 4:
                        bot.send_message(call.message.chat.id,
                                         f'Вот это да! Ты случайно не будущий профессор Дамблдор? '
                                         f'Твой результат {sum(grade)} баллов из 5. Поздравляю!')
                        bot.send_sticker(call.message.chat.id,
                                         "CAACAgIAAxkBAAELy41mAnE_9XVXy7GBxJwP8Of2MgH3QAACI0EAAkFbGUiWAZVBHkcZmjQE")
                    for i in range(len(u_answers)):
                        u_answers[i] = 0

                def one(message):
                    u_answers[0] = message.text

                @bot.message_handler(func=lambda message: message.text == '/1')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на первый вопрос:')
                    bot.register_next_step_handler(mesg, one)

                def two(message):
                    u_answers[1] = message.text

                @bot.message_handler(func=lambda message: message.text == '/2')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на второй вопрос:')
                    bot.register_next_step_handler(mesg, two)

                def three(message):
                    u_answers[2] = message.text

                @bot.message_handler(func=lambda message: message.text == '/3')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на третий вопрос:')
                    bot.register_next_step_handler(mesg, three)

                def four(message):
                    u_answers[3] = message.text

                @bot.message_handler(func=lambda message: message.text == '/4')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на четвертый вопрос:')
                    bot.register_next_step_handler(mesg, four)

                def preprocess_text(text):
                    tokens = text.lower().split()
                    tokens = [token.strip('.,?!«»"\'') for token in tokens if token not in stop_words]
                    morph = MorphAnalyzer()
                    for i in range(len(tokens)):
                        ana = morph.parse(tokens[i])
                        first = ana[0]
                        tokens[i] = first.normal_form
                    return tokens

                def cosine_similarity(vector1, vector2):
                    dot_product = np.dot(vector1, vector2)
                    norm_vector1 = np.linalg.norm(vector1)
                    norm_vector2 = np.linalg.norm(vector2)
                    similarity = dot_product / (norm_vector1 * norm_vector2)
                    return similarity

                def check_spelling(text):
                    spell = SpellChecker(language='ru')
                    misspelled = spell.unknown(text)
                    return misspelled

                def five(message):
                    u_answers[4] = message.text

                    last_line = cur_keys[5]

                    tokens1 = preprocess_text(message.text)
                    tokens2 = preprocess_text(last_line)

                    mistakes = check_spelling(tokens1)

                    word_set = set(tokens1).union(tokens2)
                    vector1 = np.array([tokens1.count(word) for word in word_set])
                    vector2 = np.array([tokens2.count(word) for word in word_set])
                    similarity_score = cosine_similarity(vector1, vector2)

                    if similarity_score >= 0:
                        per = round(similarity_score * 100, 2)
                        mes = "Твой ответ похож на идеальный на " + str(
                            per) + "%. Это не гарантия (не)правильности, но возможно ты захочешь его поменять."
                    else:
                        per = - round(similarity_score * 100, 2)
                        mes = "Твой ответ отличается от идеального на " + str(
                            per) + "%. Это не гарантия (не)правильности, но возможно ты захочешь его поменять."
                    if len(mistakes) > 0:
                        mes += "\nДружеское предупреждение: есть некоторая вероятность, что в твоем тексте есть " \
                               "опечатки. Возможно, стоит перепроверить ответ?"
                    bot.send_message(call.message.chat.id, mes)

                @bot.message_handler(func=lambda message: message.text == '/5')
                def redirect(message):
                    mesg = bot.send_message(message.chat.id, 'Введи ответ на пятый вопрос:')
                    bot.register_next_step_handler(mesg, five)

                @bot.message_handler(func=lambda message: message.text == '/finish')
                def redirect(message):
                    check_test(conn)

        test(conn)


bot.polling(none_stop=True, interval=0)
