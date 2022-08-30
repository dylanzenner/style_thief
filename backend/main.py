import uvicorn


if __name__ == "__main__":
    uvicorn.run("app.api:app", host="54.204.0.54", port=8000, reload=True)