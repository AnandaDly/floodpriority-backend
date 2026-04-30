# Memberitahu server Azure untuk menggunakan mesin ASGI (Uvicorn) khusus FastAPI
worker_class = "uvicorn.workers.UvicornWorker"
workers = 2
timeout = 120