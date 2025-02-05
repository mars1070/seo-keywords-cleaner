import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import tempfile
import zipfile
import io
from datetime import datetime
from tqdm import tqdm
from thefuzz import fuzz
from collections import defaultdict
import time

# Récupération de la clé API depuis les secrets Streamlit
if 'OPENAI_API_KEY' in st.secrets:
    openai_api_key = st.secrets['OPENAI_API_KEY']
    os.environ["OPENAI_API_KEY"] = openai_api_key
else:
    st.error("⚠️ La clé API OpenAI n'est pas configurée. Veuillez la configurer dans les secrets Streamlit.")
    st.stop()

# Initialisation du client OpenAI
client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

# Constantes
MIN_VOLUME = 20  # Volume minimum fixé à 20
MAX_KEYWORDS = 2500  # Maximum de mots-clés fixé à 2500
BATCH_SIZE = 100

# Liste des mots-clés à exclure
EXCLUDED_WORDS = {
    # Marques d'électronique et électroménager
    'philips', 'phillips', 'samsung', 'apple', 'sony', 'lg', 'panasonic', 'xiaomi', 'huawei',
    'bosch', 'siemens', 'whirlpool', 'miele', 'dyson', 'rowenta', 'moulinex', 'seb', 'krups',
    'delonghi', 'kitchenaid', 'kenwood', 'braun', 'tefal', 'beko', 'smeg', 'brandt', 'candy',
    'hoover', 'electrolux', 'aeg', 'sharp', 'toshiba', 'lenovo', 'asus', 'acer', 'hp', 'dell',
    
    # Marques de café et thé
    'nespresso', 'senseo', 'tassimo', 'dolce gusto', 'lavazza', 'lipton', 'kusmi', 'mariage freres',
    'malongo', 'carte noire', 'l or', 'illy', 'segafredo', 'maxwell', 'twinings', 'tetley',
    
    # Marques d'alimentation
    'nestle', 'danone', 'coca cola', 'pepsi', 'ferrero', 'kelloggs', 'mars', 'heinz', 'barilla',
    'bonne maman', 'president', 'lustucru', 'panzani', 'fleury michon', 'herta', 'bonduelle',
    
    # Marques de mode et beauté
    'nike', 'adidas', 'puma', 'reebok', 'under armour', 'asics', 'new balance', 'converse',
    'zara', 'h&m', 'uniqlo', 'levis', 'gap', 'ralph lauren', 'tommy hilfiger', 'lacoste',
    'loreal', 'maybelline', 'nivea', 'garnier', 'dove', 'olay', 'clinique', 'lancome',
    
    # Marques enfants et jouets
    'lego', 'playmobil', 'fisher price', 'vtech', 'chicco', 'babybjorn', 'hasbro', 'mattel',
    'nerf', 'barbie', 'pokemon', 'bandai', 'melissa doug', 'sophie la girafe',

    # Termes commerciaux
    'pas cher', 'bon marché', 'petit prix', 'meilleur prix', 'prix bas', 'discount', 'promo',
    'promotion', 'solde', 'soldes', 'destockage', 'destock', 'outlet', 'bon plan', 'bons plans',
    'moins cher', 'économique', 'economique', 'prix cassé', 'prix casse', 'braderie', 'remise',
    'reduction', 'réduction', 'offre', 'offres', 'deal', 'deals', 'vente flash', 'black friday',
    'cyber monday', 'french days',

    # Marketplaces et sites d'achat
    'amazon', 'ebay', 'aliexpress', 'walmart', 'target', 'etsy', 'rakuten', 'costco', 'bestbuy',
    'cdiscount', 'fnac', 'darty', 'boulanger', 'leroy merlin', 'castorama', 'decathlon',
    'carrefour', 'auchan', 'leclerc', 'intermarché', 'lidl', 'aldi',

    # Boutiques US additionnelles
    'bloomingdales', 'saks fifth avenue', 'bergdorf goodman', 'neiman marcus',
    'dillards', 'belk', 'jcpenney', 'lord and taylor', 'barneys', 'anthropologie',
    'free people', 'lululemon', 'athleta', 'fabletics', 'everlane', 'madewell',
    'j crew', 'banana republic', 'ann taylor', 'lane bryant', 'torrid',
    'hot topic', 'zumiez', 'journeys', 'finish line', 'famous footwear', 'dsw',
    'bath and body works', 'yankee candle', 'pier 1', 'crate and barrel',
    'pottery barn', 'west elm', 'williams sonoma', 'sur la table', 'cabelas',
    'bass pro shops', 'rei', 'eastern mountain sports', 'duluth trading',
    'lands end', 'll bean', 'eddie bauer', 'carhartt', 'orvis', 'tractor supply',
    'ace hardware', 'menards', 'harbor freight', 'northern tool', 'rockler',
    'woodcraft', 'michaels', 'joann fabrics', 'hobby lobby', 'blick art',
    'jerry arterial', 'sweetwater', 'musicians friend', 'american musical supply',

    # Boutiques allemandes
    'otto', 'zalando', 'mediamarkt', 'conrad', 'notebooksbilliger',
    'alternate', 'cyberport', 'computeruniverse', 'mindfactory', 'caseking',
    'kaufland', 'real', 'galeria', 'karstadt', 'douglas', 'dm', 'rossmann',
    'mueller', 'deichmann', 'görtz', 'about you', 'bonprix', 'tchibo', 'lidl',
    'aldi', 'rewe', 'edeka24', 'hornbach', 'obi', 'bauhaus', 'toom', 'hagebau',
    'globus', 'hellweg', 'poco', 'roller', 'xxxlutz', 'moebel', 'home24',
    'westfalia', 'klingel', 'baur', 'schwab', 'neckermann', 'weltbild',
    'thalia', 'hugendubel', 'buecher', 'medimops', 'rebuy', 'momox',
    'sport scheck', 'bike discount', 'bike components', 'fahrrad',
    'boc24', 'breuninger', 'peek und cloppenburg', 'c&a', 'ernsting family',
    'kik', 'takko', 'new yorker', 'snipes', 'foot locker', 'sidestep',
    'engelhorn', 'sportscheck', 'intersport', 'decathlon', 'pearl', 'voelkner',
    'reichelt', 'digitalo', 'pollin', 'thomann', 'music store', 'justmusic',
    'session', 'zooplus', 'fressnapf', 'futterhaus', 'medpets', 'bitiba',
    'tiierisch', 'hundeland', 'petshop',

    # Mots interrogatifs et recherche
    'how', 'what', 'where', 'when', 'why', 'which', 'who', 'whose', 'whom',
    'comment', 'pourquoi', 'quand', 'quel', 'quelle', 'quoi', 'où', 'combien', 'est ce que',
    'can i', 'should i', 'could i', 'would', 'does', 'do i need', 'what if',
    'puis je', 'devrais je', 'est il', 'faut il', 'peut on',

    # Termes commerciaux et prix
    'sale', 'discount', 'deal', 'cheap', 'free', 'best', 'buy', 'price', 'cost', 'worth',
    'solde', 'promo', 'promotion', 'prix', 'pas cher', 'gratuit', 'meilleur prix', 'bon plan',
    'clearance', 'bargain', 'offer', 'coupon', 'voucher', 'rebate', 'saving',
    'destockage', 'remise', 'reduction', 'moins cher', 'economique', 'bon marché',

    # Termes de recherche et comparaison
    'vs', 'versus', 'review', 'compare', 'comparison', 'difference between', 'better than',
    'avis', 'comparatif', 'test', 'différence entre', 'mieux que', 'plutôt que',
    'rating', 'top', 'best', 'worst', 'ranking', 'rated', 'recommended',
    'classement', 'meilleur', 'pire', 'recommandé', 'conseillé',

    # Termes informatifs et tutoriels
    'guide', 'tutorial', 'tuto', 'how to', 'tips', 'advice', 'help', 'instruction', 'manual',
    'guide', 'tutoriel', 'conseil', 'astuce', 'aide', 'mode d emploi', 'notice',
    'learn', 'explain', 'understand', 'meaning', 'definition', 'example', 'difference',
    'apprendre', 'expliquer', 'pattern', 'signification', 'définition', 'exemple',

    # Termes géographiques et localisation
    'near me', 'nearby', 'location', 'store', 'shop', 'where to buy', 'where to find',
    'près de moi', 'pres de moi', 'magasin', 'boutique', 'où acheter', 'ou acheter',
    'où trouver', 'ou trouver', 'stock', 'disponible', 'disponibilité', 'disponibilite',

    # Questions et recherches
    'what is', 'how to', 'when to', 'where to', 'why is', 'can i', 'should i',
    'quest ce que', 'qu est ce que', 'comment', 'quand', 'où', 'ou', 'pourquoi',
    'puis je', 'dois je', 'faut il',

    # Termes temporels
    'today', 'tonight', 'tomorrow', 'yesterday', 'now', 'current', 'latest',
    'aujourd hui', 'ce soir', 'demain', 'hier', 'maintenant', 'actuel', 'dernier',
    'nouvelle collection', 'new collection', 'new arrival', 'nouveauté', 'nouveaute',

    # Termes de contrefaçon
    'replica', 'replique', 'réplique', 'fake', 'faux', 'copie', 'imitation', 
    'contrefaçon', 'contrefacon', 'pas original', 'non original',

    # Autres termes non-produit
    'pdf', 'download', 'télécharger', 'telecharger', 'gratuit', 'free', 'sample',
    'échantillon', 'echantillon', 'win', 'gagner', 'contest', 'concours',
    'sale', 'vente', 'liquidation', 'destockage', 'destock', 'occasion', 'used',
    'second hand', 'seconde main', 'reconditionné', 'reconditionne',
    

}

def contains_excluded_word(keyword):
    """
    Vérifie si le mot-clé contient un des mots exclus
    """
    keyword_lower = keyword.lower()
    return any(word in keyword_lower.split() for word in EXCLUDED_WORDS)

def find_similar_keywords(keywords_df):
    """
    Version simplifiée : compare juste les mots triés
    """
    # Crée une colonne avec les mots triés
    keywords_df['Sorted_Words'] = keywords_df['Keyword'].apply(
        lambda x: ' '.join(sorted(x.lower().split()))
    )
    
    # Groupe par mots triés et garde celui avec le plus gros volume
    grouped = keywords_df.groupby('Sorted_Words').agg({
        'Keyword': 'first',  # Garde le premier mot-clé
        'Volume': 'sum',     # Additionne les volumes
    }).reset_index()
    
    # Formate le résultat
    return pd.DataFrame({
        'Keyword': grouped['Keyword'],
        'Volume': grouped['Volume'],
        'Original_Keywords': 1
    })

def clean_keywords_with_gpt(keywords_batch, current_batch_display, token_counter):
    try:
        # Afficher uniquement la liste des mots-clés
        current_batch_display.text("\n".join(keywords_batch))

        system_prompt = """Tu es un expert en SEO spécialisé dans l'optimisation de mots-clés pour le dropshipping sans marque. Ta mission est de nettoyer une liste de mots-clés en respectant ces règles STRICTES :

1. Suppression des mots-clés interdits :
Supprime tout mot-clé contenant :

Des noms de marques (ex : Nike, Samsung, Apple, Xiaomi).
Des noms de licences (ex : Disney, Marvel, Star Wars).
Des noms de marketplaces (ex : Amazon, eBay, AliExpress, Temu).
Des mots liés aux prix et promotions (ex : pas cher, gratuit, offre spéciale).
Des termes informationnels (ex : avis, review, tutoriel, guide).
Plus de 6 mots dans un mot-clé.

2. Élimination des doublons et optimisation SEO :
Garde uniquement la meilleure version des mots-clés similaires.
Choisis la formulation la plus naturelle (ex : "Table de Jardin" au lieu de "Table Jardin").
Privilégie le singulier sauf si le pluriel est plus naturel.
Corrige l'ordre des mots si nécessaire pour respecter la langue.

3. Respect strict de la langue :
Ne traduis JAMAIS les mots-clés. Garde-les strictement dans leur langue d'origine.
N'ajoute ni ne modifie aucun mot-clé.

4. RÈGLE ABSOLUE des Majuscules :
- CHAQUE mot significatif DOIT commencer par une majuscule :
  * Noms : "Table", "Chaise", "Meuble"
  * Adjectifs : "Grand", "Petit", "Rouge"
  * Matériaux : "Bois", "Verre", "Métal"
  * Couleurs : "Bleu", "Noir", "Blanc"
- Les mots de liaison restent en minuscules :
  * Articles : "le", "la", "les"
  * Prépositions : "de", "du", "des", "en", "à"
  * Conjonctions : "et", "ou"

5. Organisation des résultats :
Le premier mot-clé retourné doit être celui correspondant au terme principal de la catégorie analysée (exemple : si la liste contient "Plush Shark", alors "Plush Shark" doit être le premier résultat).
Retourne uniquement la liste finale, un mot-clé par ligne, sans commentaires ni numérotation."""

        prompt = f"Tu es un expert du Dropshipping SEO et e-commerce sans marques. Analyse ce lot de mots-clés en respectant les règles suivantes :\n" + "\n".join(keywords_batch)
        
        # Ajouter une pause plus longue pour GPT
        time.sleep(2)
        
        try:
            # Appel à l'API OpenAI
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # Mettre à jour le compteur de tokens
            token_counter['input_tokens'] += response.usage.prompt_tokens
            token_counter['output_tokens'] += response.usage.completion_tokens
            token_counter['total_tokens'] += response.usage.total_tokens
            
            # Extraire et nettoyer les mots-clés de la réponse
            cleaned_keywords = response.choices[0].message.content.strip().split('\n')
            cleaned_keywords = [kw.strip() for kw in cleaned_keywords if kw.strip()]
            cleaned_keywords = list(dict.fromkeys(cleaned_keywords))

            current_batch_display.text("Mots-clés nettoyés :")
            current_batch_display.text("\n".join(cleaned_keywords))
            
            return cleaned_keywords
        except Exception as api_error:
            error_message = str(api_error).lower()
            if "rate limit" in error_message:
                st.error(" Limite de l'API atteinte. Attendez quelques minutes avant de réessayer.")
            elif "insufficient_quota" in error_message or "billing" in error_message:
                st.error(" Crédit OpenAI épuisé ! Vous devez recharger votre compte OpenAI.")
            elif "invalid_api_key" in error_message:
                st.error(" Clé API OpenAI invalide ! Vérifiez votre fichier .env")
            else:
                st.error(f" Erreur OpenAI : {str(api_error)}")
            return []

    except Exception as e:
        st.error(f"Erreur : {str(e)}")
        return []

def create_zip_file(files_data):
    """
    Crée un fichier ZIP contenant les fichiers nettoyés
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_name, file_content in files_data:
            zip_file.writestr(file_name, file_content)
    
    zip_buffer.seek(0)
    return zip_buffer

def create_formatted_excel(merged_df):
    """
    Crée un fichier Excel avec la mise en forme demandée
    """
    try:
        # Créer un writer Excel
        output = io.BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        
        # Renommer la colonne si c'est un fichier unique
        if len(merged_df.columns) == 1:
            merged_df.columns = ['Collection 1']
        
        # Écrire le DataFrame dans Excel
        merged_df.to_excel(writer, sheet_name='Keywords', index=False)
        
        # Récupérer le workbook et la worksheet
        workbook = writer.book
        worksheet = writer.sheets['Keywords']
        
        # Définir les formats
        header_format = workbook.add_format({
            'bg_color': '#00ffff',  # Cyan pour les en-têtes
            'bold': True,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        first_row_format = workbook.add_format({
            'bg_color': '#ffff00',  # Jaune pour la première ligne
            'bold': True
        })
        
        # Appliquer le format aux en-têtes
        for col_num, value in enumerate(merged_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Appliquer le format jaune et gras à la première ligne de données
        if not merged_df.empty:
            for col_num in range(len(merged_df.columns)):
                worksheet.write(1, col_num, merged_df.iloc[0, col_num], first_row_format)
        
        # Ajuster la largeur des colonnes
        for i, col in enumerate(merged_df.columns):
            max_length = max(
                merged_df[col].astype(str).apply(len).max(),
                len(str(col))
            )
            worksheet.set_column(i, i, max_length + 2)
        
        # Sauvegarder le fichier
        writer.close()
        
        # Préparer le fichier pour le téléchargement
        output.seek(0)
        
        # Si c'est un seul fichier, utiliser le premier mot-clé comme nom
        if len(merged_df.columns) == 1:
            # Récupérer le premier mot-clé
            first_keyword = merged_df.iloc[0, 0] if not merged_df.empty else "keywords"
            # Nettoyer le nom pour un nom de fichier valide
            first_keyword = "".join(c for c in first_keyword if c.isalnum() or c in (' ', '-', '_')).strip()
            first_keyword = first_keyword.replace(' ', '_')
            filename = f"{first_keyword}_cleaned.xlsx"
        else:
            # Si multiple fichiers, garder le nom par défaut
            filename = "cleaned_keywords.xlsx"
            
        return output, filename
        
    except Exception as e:
        st.error(f"Erreur lors de la création du fichier Excel : {str(e)}")
        return None, None

def merge_cleaned_files(cleaned_dataframes):
    """
    Fusionne tous les fichiers nettoyés en un seul DataFrame
    Chaque colonne est nommée "Collection X"
    """
    # Créer un DataFrame vide
    merged_df = pd.DataFrame()
    
    # Ajouter chaque colonne
    for idx, df in enumerate(cleaned_dataframes, 1):
        if not df.empty:
            # Renommer la colonne
            df.columns = [f'Collection {idx}']
            # Ajouter au DataFrame fusionné
            merged_df = pd.concat([merged_df, df], axis=1)
    
    return merged_df

def process_csv_file(file_path, current_file_display, current_batch_display, token_counter):
    try:
        # Lire le fichier CSV avec encodage UTF-8
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # Phase 1 : Nettoyage de base
        current_file_display.markdown(f" Phase 1 : Nettoyage de base pour {Path(file_path).name}")
        
        # Convertir la colonne de mots-clés en chaînes de caractères
        df['Keyword'] = df['Keyword'].astype(str)
        
        # Nombre initial de mots-clés
        initial_count = len(df)
        
        # Supprimer les doublons
        df = df.drop_duplicates(subset=['Keyword'], keep='first')
        duplicates_removed = initial_count - len(df)
        
        # Nettoyer les espaces uniquement, sans toucher aux accents
        df['Keyword'] = df['Keyword'].apply(lambda x: x.strip())
        
        # Supprimer les mots-clés vides
        df = df[df['Keyword'].str.len() > 0]
        st.write(f"Après suppression mots-clés vides : {len(df)}")
        
        # Supprimer les doublons exacts (sensible aux accents)
        df = df.drop_duplicates(subset=['Keyword'], keep='first')
        st.write(f"Après suppression doublons : {len(df)}")
        
        # Exclure les mots-clés qui CONTIENNENT un mot ou groupe de mots de la liste d'exclusion
        def contains_excluded(keyword):
            # Convertir en minuscules pour la comparaison
            keyword_lower = keyword.lower()
            # Séparer en mots
            keyword_words = set(keyword_lower.split())
            
            # Vérifier chaque mot exclu
            for excluded_word in EXCLUDED_WORDS:
                excluded_word_lower = excluded_word.lower()
                # Si c'est un groupe de mots (ex: "pas cher")
                if ' ' in excluded_word_lower:
                    if excluded_word_lower in keyword_lower:
                        st.write(f"Mot-clé exclu : '{keyword}' (contient le groupe de mots '{excluded_word}')")
                        return True
                # Si c'est un mot unique (ex: "ou")
                else:
                    # Vérifier si le mot exclu est un mot complet dans le mot-clé
                    if excluded_word_lower in keyword_words:
                        st.write(f"Mot-clé exclu : '{keyword}' (contient le mot exact '{excluded_word}')")
                        return True
            
            return False
        
        df = df[~df['Keyword'].apply(contains_excluded)]
        st.write(f"Après exclusion mots interdits : {len(df)}")
        
        # Supprimer les mots-clés qui ont plus de 6 mots
        df['word_count'] = df['Keyword'].str.split().str.len()
        df = df[df['word_count'] <= 6]
        df = df.drop('word_count', axis=1)
        st.write(f"Après suppression mots-clés > 6 mots : {len(df)}")
        
        # Afficher quelques exemples de mots-clés restants
        if len(df) > 0:
            st.write("Exemples de mots-clés conservés :")
            st.write(df['Keyword'].head().tolist())
        else:
            st.write("Tous les mots-clés ont été filtrés")
            
            # Afficher tous les mots-clés avant filtrage pour comprendre
            st.write("Mots-clés avant filtrage :")
            st.write(df['Keyword'].tolist())
        
        # Retourner uniquement les mots-clés nettoyés
        return df['Keyword'].tolist()
    
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier {file_path}: {str(e)}")
        return [], None

def remove_similar_keywords(keywords, threshold=0.1):
    """
    Supprime les mots-clés similaires en fonction du seuil de similarité
    """
    # Créer un dictionnaire pour stocker les mots-clés uniques
    unique_keywords = {}
    
    # Itérer sur les mots-clés
    for keyword in keywords:
        # Initialiser un indicateur pour savoir si le mot-clé est similaire
        is_similar = False
        
        # Itérer sur les mots-clés uniques
        for unique_keyword in unique_keywords:
            # Calculer la similarité entre les mots-clés
            similarity = fuzz.ratio(keyword, unique_keyword)
            
            # Si la similarité est supérieure au seuil, considérer les mots-clés comme similaires
            if similarity > threshold:
                is_similar = True
                break
        
        # Si le mot-clé n'est pas similaire, l'ajouter au dictionnaire
        if not is_similar:
            unique_keywords[keyword] = True
    
    # Retourner la liste des mots-clés uniques
    return list(unique_keywords.keys())

def clean_keywords_phase1(df):
    """Phase 1 : Nettoyage algorithmique des mots-clés"""
    try:
        # Vérifier que les colonnes requises existent
        required_columns = ['Keyword', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Le fichier doit contenir les colonnes 'Keyword' et 'Volume'")
        
        st.write(f"Nombre initial de mots-clés : {len(df)}")
        
        # Convertir la colonne Volume en numérique
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        st.write(f"Après conversion volume : {len(df)}")
        
        # Si le fichier dépasse MAX_KEYWORDS lignes, on applique la limite
        if len(df) > MAX_KEYWORDS:
            # Trier par volume décroissant et garder les MAX_KEYWORDS premiers
            df = df.sort_values('Volume', ascending=False).head(MAX_KEYWORDS)
            st.write(f"Après limite MAX_KEYWORDS : {len(df)}")
        
        # Filtrer les volumes < MIN_VOLUME
        df = df[df['Volume'] >= MIN_VOLUME]
        st.write(f"Après filtre volume minimum : {len(df)}")
        
        # Nettoyer les espaces uniquement, sans toucher aux accents
        df['Keyword'] = df['Keyword'].astype(str).apply(lambda x: x.strip())
        
        # Supprimer les mots-clés vides
        df = df[df['Keyword'].str.len() > 0]
        st.write(f"Après suppression mots-clés vides : {len(df)}")
        
        # Supprimer les doublons exacts (sensible aux accents)
        df = df.drop_duplicates(subset=['Keyword'])
        st.write(f"Après suppression doublons : {len(df)}")
        
        # Exclure les mots-clés qui CONTIENNENT un mot ou groupe de mots de la liste d'exclusion
        def contains_excluded(keyword):
            # Convertir en minuscules pour la comparaison
            keyword_lower = keyword.lower()
            # Séparer en mots
            keyword_words = set(keyword_lower.split())
            
            # Vérifier chaque mot exclu
            for excluded_word in EXCLUDED_WORDS:
                excluded_word_lower = excluded_word.lower()
                # Si c'est un groupe de mots (ex: "pas cher")
                if ' ' in excluded_word_lower:
                    if excluded_word_lower in keyword_lower:
                        st.write(f"Mot-clé exclu : '{keyword}' (contient le groupe de mots '{excluded_word}')")
                        return True
                # Si c'est un mot unique (ex: "ou")
                else:
                    # Vérifier si le mot exclu est un mot complet dans le mot-clé
                    if excluded_word_lower in keyword_words:
                        st.write(f"Mot-clé exclu : '{keyword}' (contient le mot exact '{excluded_word}')")
                        return True
            
            return False
        
        df = df[~df['Keyword'].apply(contains_excluded)]
        st.write(f"Après exclusion mots interdits : {len(df)}")
        
        # Supprimer les mots-clés qui ont plus de 6 mots
        df['word_count'] = df['Keyword'].str.split().str.len()
        df = df[df['word_count'] <= 6]
        df = df.drop('word_count', axis=1)
        st.write(f"Après suppression mots-clés > 6 mots : {len(df)}")
        
        # Afficher quelques exemples de mots-clés restants
        if len(df) > 0:
            st.write("Exemples de mots-clés conservés :")
            st.write(df['Keyword'].head().tolist())
        else:
            st.write("Tous les mots-clés ont été filtrés")
            
            # Afficher tous les mots-clés avant filtrage pour comprendre
            st.write("Mots-clés avant filtrage :")
            st.write(df['Keyword'].tolist())
        
        # Retourner uniquement les mots-clés nettoyés
        return df['Keyword'].tolist()
    
    except Exception as e:
        raise Exception(f"Erreur lors du nettoyage phase 1 : {str(e)}")

def main():
    st.title(" Nettoyeur de Mots-Clés SEO - Etape 1")
    
    # Ajouter le CSS pour l'effet pulse et les stats
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        
        .stButton button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 0.5rem !important;
            margin-top: 1rem !important;
            width: 100% !important;
        }
        
        .stDownloadButton button {
            background-color: #2196F3 !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 0.5rem !important;
            margin: 0.5rem 0 !important;
            width: 100% !important;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
            100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
        }
        
        .combined-button button {
            animation: pulse 2s infinite !important;
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            padding: 1.2rem !important;
            border-radius: 0.5rem !important;
            margin: 2rem auto !important;
            width: 100% !important;
            font-size: 1.3em !important;
            font-weight: 900 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }
        
        .stats-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-icon {
            font-size: 2em;
            color: #4CAF50;
            display: block;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        .stat-value {
            font-weight: 700;
            color: #1e3d59;
            font-size: 1.1em;
        }
        
        .progress-container {
            margin: 1rem 0;
        }
        
        .stProgress .st-bo {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialiser les variables de session si elles n'existent pas
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'all_dataframes' not in st.session_state:
        st.session_state.all_dataframes = []
    if 'token_counter' not in st.session_state:
        st.session_state.token_counter = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }
    
    # Initialiser la liste des DataFrames nettoyés dans la session
    if 'cleaned_dataframes' not in st.session_state:
        st.session_state.cleaned_dataframes = []
    
    # Zone de dépôt des fichiers
    uploaded_files = st.file_uploader(" Déposez vos fichiers Excel ou CSV", 
                                    type=['xlsx', 'csv'], 
                                    accept_multiple_files=True)

    # Créer des colonnes pour les boutons
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        if st.button(" Nettoyer les Mots-Clés", type="primary"):
            if not uploaded_files:
                st.error("Veuillez d'abord déposer des fichiers.")
                return
            
            # Réinitialiser la session
            st.session_state.cleaned_dataframes = []
            st.session_state.token_counter = {
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
                
            # Créer les conteneurs pour l'affichage du traitement
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.markdown("""
                    <div class='stats-container'>
                        <span class='stat-icon'>📊</span>
                        <div class='stat-label'>Fichier en cours</div>
                        <div id='file-progress' class='stat-value'></div>
                    </div>
                """, unsafe_allow_html=True)
                file_progress_text = st.empty()
            
            with stats_cols[1]:
                st.markdown("""
                    <div class='stats-container'>
                        <span class='stat-icon'>🔄</span>
                        <div class='stat-label'>Lot en cours</div>
                        <div id='batch-progress' class='stat-value'></div>
                    </div>
                """, unsafe_allow_html=True)
                current_batch_container = st.empty()
            
            with stats_cols[2]:
                st.markdown("""
                    <div class='stats-container'>
                        <span class='stat-icon'>💸</span>
                        <div class='stat-label'>Coût total</div>
                        <div id='cost-estimate' class='stat-value'></div>
                    </div>
                """, unsafe_allow_html=True)
                cost_container = st.empty()
            
            with stats_cols[3]:
                st.markdown("""
                    <div class='stats-container'>
                        <span class='stat-icon'>📈</span>
                        <div class='stat-label'>Progression</div>
                        <div id='progress' class='stat-value'></div>
                    </div>
                """, unsafe_allow_html=True)
                progress_container = st.empty()
            
            # Conteneur pour la barre de progression
            st.markdown("<div class='progress-container'>", unsafe_allow_html=True)
            progress_bar = progress_container.progress(0)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Traiter chaque fichier
            progress_bar = progress_container.progress(0)
            total_files = len(uploaded_files)
            
            # Liste pour stocker les résultats de la phase 1
            phase1_results = []
            
            # PHASE 1 : Traiter tous les fichiers avec le nettoyage algorithmique
            for i, file in enumerate(uploaded_files):
                # Afficher le fichier en cours
                file_progress_text.text(f"Fichier en cours : {i + 1}/{total_files}")
                
                try:
                    # Lire le fichier
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file, encoding='utf-8')
                    else:
                        df = pd.read_excel(file)
                    
                    # Phase 1 : Nettoyage algorithmique
                    cleaned_keywords_phase1 = clean_keywords_phase1(df)
                    
                    # Si aucun mot-clé ne passe la phase 1, passer au fichier suivant
                    if not cleaned_keywords_phase1:
                        st.warning(f"Aucun mot-clé valide trouvé dans {file.name} après la phase 1")
                        continue
                        
                    # Stocker les résultats de la phase 1
                    phase1_results.append({
                        'file_name': file.name,
                        'keywords': cleaned_keywords_phase1
                    })
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier {file.name}: {str(e)}")
                    continue
                
                # Mettre à jour la barre de progression
                progress_bar.progress((i + 1) / total_files)
            
            # PHASE 2 : Traiter les résultats de la phase 1 avec GPT
            all_cleaned_dfs = []
            for i, result in enumerate(phase1_results):
                file_progress_text.text(f"Fichier en cours : {i + 1}/{len(phase1_results)}")
                
                try:
                    # Phase 2 : Nettoyage GPT
                    cleaned_keywords = []
                    keywords_phase1 = result['keywords']
                    total_batches = len(keywords_phase1) // BATCH_SIZE + (1 if len(keywords_phase1) % BATCH_SIZE > 0 else 0)
                    
                    # Barre de progression pour les lots
                    batch_progress = st.progress(0)
                    
                    # Traiter les mots-clés par lots
                    for batch_num, start_idx in enumerate(range(0, len(keywords_phase1), BATCH_SIZE)):
                        # Mettre à jour la progression des lots
                        current_batch_container.text(f"Lot en cours : {batch_num + 1}/{total_batches}")
                        
                        # Préparer le lot
                        end_idx = min(start_idx + BATCH_SIZE, len(keywords_phase1))
                        current_batch = keywords_phase1[start_idx:end_idx]
                        
                        # Nettoyer le lot avec GPT
                        cleaned_batch = clean_keywords_with_gpt(current_batch, current_batch_container, st.session_state.token_counter)
                        if cleaned_batch:
                            cleaned_keywords.extend(cleaned_batch)
                        
                        # Mettre à jour la progression des lots
                        batch_progress.progress((batch_num + 1) / total_batches)
                        
                        # Petite pause entre les lots pour éviter les limites de l'API
                        time.sleep(0.5)
                    
                    # Créer le DataFrame final pour ce fichier
                    if cleaned_keywords:
                        cleaned_df = pd.DataFrame({
                            'Keyword': cleaned_keywords,
                            'Volume': [0] * len(cleaned_keywords)  # Volume par défaut
                        })
                        all_cleaned_dfs.append(cleaned_df)
                        
                        # Créer et télécharger le fichier Excel
                        excel_data, filename = create_formatted_excel(cleaned_df)
                        
                        # Ajouter le bouton de téléchargement
                        with col2:
                            st.download_button(
                                label=f" Télécharger {result['file_name'].replace('.csv', '_clean.xlsx')}",
                                data=excel_data,
                                file_name=result['file_name'].replace('.csv', '_clean.xlsx'),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                except Exception as e:
                    st.error(f"Erreur lors du traitement GPT du fichier {result['file_name']}: {str(e)}")
                    continue
                
                # Mettre à jour la barre de progression
                progress_bar.progress((i + 1) / len(phase1_results))
            
            # Créer un DataFrame combiné avec toutes les collections
            if len(all_cleaned_dfs) > 1:
                # Créer un dictionnaire pour stocker les collections
                collections_dict = {}
                
                # Traiter chaque DataFrame séparément
                for i, df in enumerate(all_cleaned_dfs, 1):
                    collections_dict[f'Collection {i}'] = df['Keyword'].tolist()
                
                # Trouver la longueur maximale
                max_len = max(len(keywords) for keywords in collections_dict.values())
                
                # Créer un nouveau DataFrame avec des colonnes alignées
                combined_data = {}
                for collection_name, keywords in collections_dict.items():
                    # Étendre la liste avec None pour avoir la même longueur
                    padded_keywords = keywords + [None] * (max_len - len(keywords))
                    combined_data[collection_name] = padded_keywords
                
                combined_df = pd.DataFrame(combined_data)
                
                # Créer le fichier Excel avec mise en forme
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    # Écrire le DataFrame principal
                    combined_df.to_excel(writer, sheet_name='Keywords', index=False)
                    
                    # Récupérer le workbook et la worksheet
                    workbook = writer.book
                    worksheet = writer.sheets['Keywords']
                    
                    # Définir les formats
                    header_format = workbook.add_format({
                        'bg_color': '#00ffff',  # Cyan pour les en-têtes
                        'bold': True,
                        'align': 'center',
                        'valign': 'vcenter'
                    })
                    
                    first_row_format = workbook.add_format({
                        'bg_color': '#ffff00',  # Jaune pour la première ligne
                        'bold': True
                    })
                    
                    # Appliquer le format aux en-têtes
                    for col_num, value in enumerate(combined_df.columns.values):
                        worksheet.write(0, col_num, value, header_format)
                    
                    # Appliquer le format jaune et gras à la première ligne de données
                    if not combined_df.empty:
                        for col_num in range(len(combined_df.columns)):
                            if pd.notna(combined_df.iloc[0, col_num]):
                                worksheet.write(1, col_num, combined_df.iloc[0, col_num], first_row_format)
                    
                    # Ajuster la largeur des colonnes
                    for i, col in enumerate(combined_df.columns):
                        max_length = max(
                            combined_df[col].astype(str).apply(len).max(),
                            len(str(col))
                        )
                        worksheet.set_column(i, i, max_length + 2)
                
                excel_buffer.seek(0)
                excel_data = excel_buffer.getvalue()
                
                # Ajouter un espacement avant le bouton combiné
                with col2:
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.markdown("<div class='combined-button'>", unsafe_allow_html=True)
                    st.download_button(
                        label=" TÉLÉCHARGER TOUTES LES COLLECTIONS COMBINÉES ",
                        data=excel_data,
                        file_name="all_collections.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_all"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Effacer les conteneurs de progression une fois terminé
            progress_container.empty()
            file_progress_text.empty()
            current_batch_container.empty()
            
            # Afficher les statistiques d'utilisation
            st.markdown("### Statistiques d'utilisation")
            token_counter = st.session_state.token_counter
            input_cost = (token_counter['input_tokens'] / 1000) * 0.0015
            output_cost = (token_counter['output_tokens'] / 1000) * 0.002
            total_cost = input_cost + output_cost
            
            cost_container.markdown(f"<div class='stats-container'>\n    <span class='stat-icon'>💸</span>\n    <div class='stat-label'>Coût total</div>\n    <div class='stat-value'>${total_cost:.4f}</div>\n</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            - Tokens en entrée : {token_counter['input_tokens']:,} (${input_cost:.4f})
            - Tokens en sortie : {token_counter['output_tokens']:,} (${output_cost:.4f})
            - Total tokens : {token_counter['total_tokens']:,}
            """)
            
            st.success("Traitement terminé !")
            
if __name__ == "__main__":
    main()