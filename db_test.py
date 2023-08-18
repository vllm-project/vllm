from dotenv import load_dotenv
import os
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
column_name = "text"

# Insert data into the database
def insert_into_db(output):
    try:
        query = f"INSERT INTO {table_name} ({column_name}) VALUES (%s)"
        cursor.execute(query, (output,))
        connection.commit()
    except Exception as e:
        print("Error inserting data into the database:", e)
        connection.rollback()

# Example usage
output = "Your output text here"
insert_into_db(output)

# Close the database connection
cursor.close()
connection.close()