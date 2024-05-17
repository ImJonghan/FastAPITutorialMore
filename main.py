from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image/'):
        # 이미지 파일 읽기
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # 이미지를 그레이스케일로 변환
        gray_image = image.convert('L')

        # 변환된 이미지를 byte로 변환
        img_byte_arr = io.BytesIO()
        gray_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # StreamingResponse로 이미지 반환
        return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="Invalid file format.")
    
@app.post("/rotate/")
async def rotate_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 이미지를 90도 회전
    rotated_image = image.rotate(90, expand=True)

    img_byte_arr = io.BytesIO()
    rotated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    return StreamingResponse(io.BytesIO(img_byte_arr), media_type="image/png")

from pydantic import BaseModel
import numpy as np
import pickle

# 모델 로드
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict_iris(iris: IrisModel):
    data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

class TextData(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(data: TextData):
    inputs = tokenizer(data.text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

        predicted_class_id = logits.argmax().item()
        model.config.id2label[predicted_class_id]
    return {"result": predicted_class_id}

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
import pandas as pd

# 데이터 로드
books = pd.read_excel('science_books.xlsx')

# 임베딩 모델 초기화
sbert = SentenceTransformerEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# 벡터 저장소 생성
vector_store = Chroma.from_texts(
    texts=books['제목'].tolist(),
    embedding=sbert
)

@app.post("/search/")
def search_books(query: str):
    results = vector_store.similarity_search(query=query, k=3)  # 상위 3개 결과 반환
    return {"query": query, "results": results}
