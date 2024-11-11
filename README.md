# PDF-ChatBot
PDF Chatbot is a Streamlit-based application that allows users to chat with a conversational AI model trained on PDF documents. The chatbot extracts information from uploaded PDF files and answers user questions based on the provided context.


## Dockeraized Development

If you have docker installed, you can run the application using the following command:

- Obtain a Google API key and set it in the `.env` file.

   ```.env
   GOOGLE_API_KEY=your_api_key_here
   ```

```bash
docker build -t pdf_chatbot .
docker run -it -p 8501:8501 pdf_chatbot bash
streamlit run pdf_chatbot.py

```

Your application will be available at <http://localhost:8501>.

## Local Development

Follow these instructions to set up and run this project on your local machine.

   **Note:** This project requires Python 3.10 or higher.

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Ismail-Ryad/PDF-ChatBot.git

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google API Key:**
   - Obtain a Google API key and set it in the `.env` file.

   ```bash
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the Application:**

   ```bash
   streamlit run pdf_chatbot.py
   ```

## Chatbot Fearues & Functionalities

**Upload Documents Interface**
   - Use the sidebar to upload PDF files.
   - Click on "Submit" to extract text and generate embeddings.
      
**Chat Interface:**
   - Chat with the AI in the main interface [Ask questions and get answers from Docs uploaded]

## Project Structure

- `pdf_chatbot.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## API Documentation
Google Gemini: For providing the language model.

Streamlit: the user interface framework.

Langchain Framework
