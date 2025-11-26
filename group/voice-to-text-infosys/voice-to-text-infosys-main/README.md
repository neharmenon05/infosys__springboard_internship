1.Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run the app
uvicorn app:app --reload or if access is denied you can use python app.py


4. open in browser
http://localhost:8000



