import os
from typing import List, Tuple
import sqlite3
import pandas as pd


class ProcessFiles:
    """
    A class to process uploaded files, converting them to a SQL database format.

    This class handles both CSV and XLSX files, reading them into pandas DataFrames and
    storing each as a separate table in the SQL database specified by the application configuration.
    """
    def __init__(self, files_dir: str, chatbot: List) -> None:
        """
        Initialize the ProcessFiles instance.

        Args:
            files_dir (List): A list containing the file paths of uploaded files.
            chatbot (List): A list representing the chatbot's conversation history.
        """
        self.files_dir = files_dir
        self.chatbot = chatbot

    def _process_uploaded_files(self) -> Tuple:
        """
        Private method to process the uploaded files and store them into the SQL database.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        for file_dir in self.files_dir:
            print("self.files_dir = ",file_dir)
            file_names_with_extensions = os.path.basename(file_dir)
            file_name, file_extension = os.path.splitext(
                file_names_with_extensions)
            print("file_name = ", file_name)
            print("file_extension = ",file_extension)
            if file_extension == ".csv":
                df = pd.read_csv(file_dir)
            elif file_extension == ".xlsx":
                df = pd.read_excel(file_dir)
            else:
                raise ValueError("The selected file type is not supported")
            # Connect to SQLite database
            conn = sqlite3.connect('uploaded_csv_sql.db')
            cursor = conn.cursor()
            df.to_sql("table_name", conn, if_exists="replace", index=False)
            conn.commit()
            conn.close()

            #df.to_sql(file_name, self.engine, index=False)
        print("==============================")
        print("All csv/xlsx files are saved into the sql database.")
        self.chatbot.append(
            (file_name+file_extension, "Uploaded files are ready. Please ask your question"))
        return "", self.chatbot


    def run(self):
        """
        public method to execute the file processing pipeline.

        Includes steps for processing uploaded files and validating the database.

        Returns:
            Tuple[str, List]: A tuple containing an empty string and the updated chatbot conversation list.
        """
        input_txt, chatbot = self._process_uploaded_files()
        return input_txt, chatbot


class UploadFile:
    @staticmethod
    def run_pipeline(files_dir: str, chatbot: List):
        """
        Run the appropriate pipeline based on chatbot functionality.

        Args:
            files_dir (List): List of paths to uploaded files.
            chatbot (List): The current state of the chatbot's dialogue.
            chatbot_functionality (str): A string specifying the chatbot's current functionality.

        Returns:
            Tuple: A tuple of an empty string and the updated chatbot list, or None if functionality not matched.
        """
        input_txt, chatbot = ProcessFiles(
                files_dir=files_dir, chatbot=chatbot).run()
        return input_txt, chatbot