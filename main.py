import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from transformers import pipeline

# ==============================
# Initialisation de l'application FastAPI
# ==============================
app = FastAPI()

# Configuration du logging pour debug
logging.getLogger().setLevel(logging.INFO)

# Gestion des templates pour la page d'accueil
templates = Jinja2Templates(directory="templates")

# ==============================
# Chargement des modèles et vectorizers
# ==============================
try:
    cv = joblib.load("models/ft_title_N_cv.pkl")
    tfidf = joblib.load("models/ft_title_N_tfidf.pkl")
    model_supervise = joblib.load("models/model_supervise.pkl")
    lda_model = joblib.load("models/best_lda_model_ft_title_N.pkl")
    mlb = joblib.load("models/mlb.pkl")  # MultiLabelBinarizer pour les tags supervisés
    topic_keywords = joblib.load("models/lda_topic_keywords.pkl")  # Mots-clés LDA
    nlp_bert = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    logging.info("✅ Tous les modèles et vectorizers ont été chargés avec succès.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des modèles : {str(e)}")
    raise HTTPException(status_code=500, detail="Impossible de charger les modèles.")

# ==============================
# Fonctions Utilitaires
# ==============================
def decode_supervised_prediction(binary_output):
    """Convertit la sortie du modèle supervisé en liste de tags."""
    try:
        return mlb.inverse_transform(binary_output)[0]  # Liste des tags
    except Exception as e:
        logging.error(f"Erreur lors du décodage des tags supervisés : {e}")
        return ["Erreur décodage"]

def get_lda_keywords(topic_distribution):
    """Retourne les mots-clés du topic dominant en LDA."""
    try:
        dominant_topic = np.argmax(topic_distribution)
        return topic_keywords.get(dominant_topic, ["Aucun mot-clé trouvé"])
    except Exception as e:
        logging.error(f"Erreur lors de l'extraction des topics LDA : {e}")
        return ["Erreur LDA"]

# ==============================
# Modèle de requête Pydantic
# ==============================
class QuestionRequest(BaseModel):
    question: str

# ==============================
# Page d'accueil (index.html)
# ==============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ==============================
# Endpoint de prédiction des tags
# ==============================
@app.post("/predict-tags/")
async def predict_tags(request: QuestionRequest):
    question = request.question.strip()  # Supprimer les espaces inutiles

    if not question:
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")

    try:
        # 🔹 Vectorisation du texte
        X_cv = cv.transform([question])
        X_tfidf = tfidf.transform(X_cv)

        # 🔹 Prédiction supervisée (convertir 0/1 en mots-clés)
        supervised_prediction_binary = model_supervise.predict(X_tfidf)
        supervised_tags = decode_supervised_prediction(supervised_prediction_binary)

        # 🔹 Prédiction LDA (associer topics à tags)
        lda_topic_distribution = lda_model.transform(X_tfidf)[0]
        lda_tags = get_lda_keywords(lda_topic_distribution)

        # 🔹 Prédiction BERT Zero-shot (analyse dynamique des tags)
        all_tags = mlb.classes_.tolist()  # Liste complète des tags connus
        bert_result = nlp_bert(question, candidate_labels=all_tags)

        # 🔹 Construire la réponse finale
        response = {
            "supervised_tags": supervised_tags,
            "lda_topics": lda_tags,
            "bert_tags": bert_result["labels"][:10]
        }

        logging.info(f"Réponse API : {response}")
        return response

    except Exception as e:
        logging.error(f"Erreur API : {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")
