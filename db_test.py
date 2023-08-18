from supabase import create_client, Client

from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()

# Get the URL and KEY from environment variables
database_url = os.getenv('database_url')
database_key = os.getenv('database_key')


supabase: Client = create_client(database_url, database_key)

data, count = supabase.table('test').insert({"text": "Denmark"}).execute()