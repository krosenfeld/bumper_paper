""" threadb.py
Class for managing threads in a SQLite database.
"""

import sqlite3
from openai import OpenAI

class Threadb:
    def __init__(self, db_name='threads.db'):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """
        Create the database table if it doesn't exist.
        """
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY,
            thread_id TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()

    def add_thread(self, thread_id):
        """
        Add a new thread to the database.
        """
        self.cursor.execute('''
        INSERT INTO threads (thread_id)
        VALUES (?)
        ''', (thread_id,))
        self.conn.commit()

    def get_all_threads(self):
        """
        Retrieve all threads from the database.
        """
        self.cursor.execute('SELECT * FROM threads')
        return self.cursor.fetchall()

    def find_thread_by_id(self, thread_id):
        """
        Find a specific thread by its ID.
        """
        self.cursor.execute('SELECT * FROM threads WHERE thread_id = ?', (thread_id,))
        return self.cursor.fetchone()

    def remove_thread_by_id(self, thread_id):
        """
        Remove a specific thread by its ID.
        """
        self.cursor.execute('DELETE FROM threads WHERE thread_id = ?', (thread_id,))
        self.conn.commit()
        response = OpenAI().beta.threads.delete(thread_id)
        print(response)

    def delete_all_threads(self):
        """
        Delete all existing threads from the database.
        """
        threads = self.get_all_threads()
        for thread in threads:
            thread_id = thread[1]
            self.remove_thread_by_id(thread_id)

    def close(self):
        """
        Close the database connection.
        """
        self.conn.close()

    def __del__(self):
        """
        Destructor method that is called when Threadb object is deleted.
        """
        self.close()