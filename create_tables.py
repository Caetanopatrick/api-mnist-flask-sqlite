import sqlite3

connection = sqlite3.connect('data.db')
cursor = connection.cursor()

create_table = "CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, time datetime, prediction text)"
cursor.execute(create_table)


connection.close()
