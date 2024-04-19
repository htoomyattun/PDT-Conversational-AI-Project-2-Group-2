import sqlite3


def insert_data(a, b, c, d,e,f,g,h):
    conn = sqlite3.connect('studentdemo.db')
    cursor = conn.cursor()


    cursor.execute(
        '''INSERT INTO test(module1, module2, module3, module4,favmodule,leastfavmodule, intdomain, notintdomain ) VALUES (?, ?, ?, ?,?,?,?,?)''',
        (a, b, c, d,e,f,g,h,))


    conn.commit()

    conn.close()
