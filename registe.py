import sqlite3
import os

def list_registered_users():
    # Path to the SQLite database under 'instance/' directory
    db_path = os.path.join("instance", "users.db")
    
    # Check if the database file exists
    if not os.path.exists(db_path):
        print("Error: Database file 'users.db' not found in the 'instance' folder.")
        return
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query to select name, email, and password from the User table
        cursor.execute("SELECT name, email, password FROM User")
        users = cursor.fetchall()
        
        # Check if any users are found
        if not users:
            print("No registered users found.")
        else:
            print("Registered Users:")
            for name, email, password in users:
                print(f"- Name: {name}, Email: {email}, Password: {password}")
        
        # Close the connection
        conn.close()
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    list_registered_users()
