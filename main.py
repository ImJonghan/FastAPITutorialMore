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