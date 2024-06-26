Бот предназначается для фанатов Гарри Поттера, которые хотели бы попробовать пройти экзамены по предметам Хогвартса. В боте предлагается 2 предмета на выбор: Чары и Зельеварение. После нажатия команды /start выводится приветственное сообщение, которое предлагает выбрать предмет (для каждого предмета есть своя кнопка). После этого выводится сообщение с описанием правил прохождения экзамена (как это работает) и кнопка "Начать". Когда пользователь нажимает на нее, ему вылезает список из 5 вопросов с соответствующими командами: вы нажимаете на команду с номером вопроса, на который хотите ответить, и вас просят ввести ответ. Когда вы готовы сдать ответы, вы нажимаете на команду /finish. Тогда бот подсчитывает результаты и выдает количество баллов, напутственное сообщение и стикер. После этого можно заново нажать на /start и выбрать другой предмет или пройти этот же заново. Список из 5 вопросов рандомно выбирается из таблиц соответствующего предмета в базе данных, где хранится 20 вопросов с выбором ответа и 5 "эссейных" вопросов с развернутым ответом (для каждого предмета).

Кроме того, есть команды: /stats и /help. Команда /help выдает описание того, как работает бот, а команда /stats - столбчатую диаграмму со статистикой по прохождениям (сколько людей - вернее, прохождений - получили какую оценку). К командам /start, /stats и /help всегда можно обратиться по встроенной кнопке Menu рядом со скрепочкой.

Что (отдаленно) лингвистического я там использую:
- Во-первых, автоматическая проверка развернутых ответов: ответ токенизируется, очищается и слова приводятся к начальной форме, что потом сравнивается со списком слов, вбитым в ключи
- Во-вторых, еще на этапе сдачи ответа на пятый вопрос (тот самый вопрос с развернутым ответом) проверяется косинусная близость между сданным ответом и "идеальным" примером ответа. Здесь я, к сожалению, не смогла использовать нормальную семантическую модель (пайчарм просто отказался их все устанавливать), поэтому, к сожалению, именно семантическая близость не учитывается, а используется старый добрый мешок слов (естественно, слова сначала предобрабатываются, из них удаляются стоп-слова)
- В-третьих, я проверяю текст пользователя на возможные опечатки с помощью spellchecker: он далеко не идеален (как минимум, для русского), но тем не менее может относительно неплохо находить неправильно написанные слова, от которых может зависеть результат теста. Такое ощущение, что он иногда считает формы слов неправильным написанием, поэтому там тоже есть приведение к начальному виду

Мой pythonanywhere: sugarplum.pythonanywhere.com

Сам бот: https://t.me/hp_hogwarts_exams_bot

Requirements:

Вам понадобятся следующие библиотеки и модули:
- import telebot
- from telebot import types
- import sqlite3
- from pymorphy2 import MorphAnalyzer
- import re
- import numpy as np
- import matplotlib.pyplot as plt
- from io import BytesIO
- import nltk
- from nltk.corpus import stopwords
- from spellchecker import SpellChecker
