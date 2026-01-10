# superfacial

## Runing the App 
This project uses a simple static frontend and a Flask backend API. 

1. Start the backend (Flask)

```
conda activate sf
cd backend
python app.py
``` 
The API will run at 
https://127.0.0.1:8000 

2. Start frontend (Static server)
From the project root: 
```
python -m http.server 5500
```
Open in your browser:
```
http://localhost:5500/app/
```
How it works

- The frontend (HTML/CSS/JS) collects user labels via an interactive interface.

- On save, selections are sent to the backend (POST /save).

- The backend stores labels as CSV and trains a preference model using face embeddings.

- The trained model can then be used for inference on new images.