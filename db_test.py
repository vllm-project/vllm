from dotenv import load_dotenv
import os
import json
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Replace [YOUR-PASSWORD] with your actual password
db_config = {
    "user": "postgres",
    "password": os.getenv('db-pwd'),
    "host": os.getenv('db-host'),
    "port": os.getenv('db-port'),
    "database": "postgres",
}

# Connect to the database
connection = psycopg2.connect(**db_config)
cursor = connection.cursor()

# Define the table name and column name
table_name = "test"
column_name1 = "text"
column_name2 = "type"

# Insert data into the database
def insert_into_db(action, type):
    try:
        query = f"INSERT INTO {table_name} ({column_name1}, {column_name2}) VALUES (%s, %s)"
        cursor.execute(query, (action, type))
        connection.commit()
    except Exception as e:
        print("Error inserting data into the database:", e)
        connection.rollback()

# Example usage

prompt_answer_log = {
    "prompt": "This is the prompt",
    "answer": "OUTPUT"
}

insert_into_db(json.dumps(prompt_answer_log), "prompt_answer")

# Close the database connection
cursor.close()
connection.close()