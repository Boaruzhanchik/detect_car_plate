import sqlite3

connection = sqlite3.connect('vehicle_numbers.db')
cursor = connection.cursor()

success = True
for number in ['A695KA799', 'C065MK78', 'A888AA00', 'H778EM799', 'K425CE82', 'H961BB82']:
    try:
        cursor.execute('INSERT INTO allowed_numbers (number) VALUES (?)', (number,))
    except sqlite3.Error:
        success = False
        break

connection.commit()
connection.close()

if success:
    print('Данные загружены успешно')
else:
    print('Ошибка при загрузке данных')


#import sqlite3
#
#connection = sqlite3.connect('vehicle_numbers.db')
#cursor = connection.cursor()
#
#with open('vehicle_numbers.txt', 'r') as f:
#    for line in f:
#        number = line.strip()
#        try:
#            cursor.execute('INSERT INTO allowed_numbers (number) VALUES (?)', (number,))
#        except sqlite3.Error:
#            success = False
#            break
#
#connection.commit()
#connection.close()
#
#if success:
#    print('Данные загружены успешно')
#else:
#    print('Ошибка при загрузке данных')