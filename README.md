# csaiml-chatbot

Now to Start This Project 

1. First clone the repository on your system
    git clone

2. no wdownload the requirements.txt file
   pip install -r requirements.txt

3. If you would like to rebuild the faiss and perform the data extraction then run these commands on your terminal, else move to step 4
     python extract_pdf_data.py
     python clean_and_chunk.py
     python embed_and_index.py

4. now first check the running of the llm
    python llm_integration.py

5. Now to fully run on the terminal window
     python chatbot_window.py
