from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from routers import apis


middleware = [Middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])]

app = FastAPI(middleware=middleware)
app.include_router(apis.router)