from fastapi import FastAPI

app = FastAPI(
    title="ADHD Chatbot API",
    description="An AI-powered ADHD psychoeducation assistant",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "ADHD Chatbot API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}