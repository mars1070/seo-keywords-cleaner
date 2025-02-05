# SEO Keywords Cleaner

Application Streamlit pour nettoyer en masse les mots-clés provenant de fichiers SEMrush en utilisant GPT-3.5.

## Installation

1. Cloner le projet
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Créer un fichier `.env` à la racine du projet et ajouter votre clé API OpenAI :
```
OPENAI_API_KEY=votre_clé_api
```

## Utilisation

1. Lancer l'application :
```bash
streamlit run app.py
```

2. Uploader vos fichiers CSV SEMrush (jusqu'à 1000 fichiers)
3. Cliquer sur "Lancer le nettoyage"
4. Télécharger le fichier final nettoyé

## Fonctionnalités

- Upload massif de fichiers CSV (jusqu'à 1000 fichiers)
- Nettoyage intelligent des mots-clés via GPT-3.5
- Traitement par lots de 100 mots-clés
- Fusion automatique des résultats
- Interface utilisateur intuitive
