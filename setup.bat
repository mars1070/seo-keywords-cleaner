@echo off
echo Installation de SEO Keywords Cleaner...

REM Création de l'environnement virtuel
python -m venv venv

REM Activation de l'environnement virtuel
call venv\Scripts\activate.bat

REM Installation des dépendances
pip install --upgrade pip
pip install streamlit==1.29.0
pip install openai==1.3.7
pip install python-dotenv==1.0.0
pip install tqdm==4.66.1
pip install pandas==2.2.0

echo Installation terminée !
echo.
echo Pour lancer l'application :
echo 1. Créez un fichier .env avec votre clé API OpenAI
echo 2. Double-cliquez sur "Launch SEO Keywords Cleaner.vbs"
pause
