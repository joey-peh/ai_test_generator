Start virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Stop:
deactivate

Run streamlit:
streamlit run app.py

Run local deepseek cmd:
brew services start ollama
ollama run deepseek-coder
ollama run qwen2.5-coder# ai_test_generator
