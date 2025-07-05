import os
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

qa_chain = None
llm = None

class QuestionRequest(BaseModel):
    question: str

def initialize_chatbot():
    global qa_chain, llm
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.1, 
        max_tokens=300,
        google_api_key=google_api_key
    )
    loader = PyPDFLoader("teja.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=150,  
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)
    prompt = PromptTemplate(
        template="""You are answering questions about Suryateja's profile. Use the provided context to answer the question accurately and comprehensively.

        INSTRUCTIONS:
        - Search through ALL the provided context carefully
        - Look for direct mentions, related terms, and implicit references
        - If you find relevant information, provide a complete answer
        - If the exact information isn't available but related information exists, mention what is available
        - Only say "Information not available" if there's truly no relevant information in the context
        
        Context: {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 6,  
                "fetch_k": 20,  
                "lambda_mult": 0.7  
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def generate_questions(context: str, current_answer: str) -> List[str]:
    response = llm.invoke([{"role": "user", "content": 
        f"""Based on the following context about Suryateja and the current answer provided, generate 3 specific, factual follow-up questions that are related to the current answer and can be answered from the context. The questions should naturally flow from the current conversation.

Context: {context[:1000]}

Current Answer: {current_answer}

Generate 3 related follow-up questions (each ending with '?') that would naturally come next in this conversation:
Note: 
Ensure the questions are specific and can be answered with the provided context. 
if there is no change of question related to the current answer, then generate questions based on the context.just change the question to be related to the context.
remember the questions are also answered by chat bot, not by the user.
"""
    }])
    
    questions = [line.strip() for line in response.content.split('\n') 
                if '?' in line and len(line.strip()) > 10][:3]
    
    if not questions:
        questions = [
            "What programming languages does Suryateja know?",
            "Which university is Suryateja studying at?",
            "What is Suryateja's current CGPA?"
        ]
    
    return questions

@app.on_event("startup")
async def startup_event():
    initialize_chatbot()

@app.post("/chat")
async def chat(request: QuestionRequest):
    if qa_chain is None:
        return {"error": "Chatbot not initialized"}
    
    result = qa_chain.invoke({"query": request.question})
    answer = result["result"]
    
    context = " ".join([doc.page_content for doc in result["source_documents"][:4]])
    questions = generate_questions(context, answer)
    
    return {"result": [answer] + questions}

@app.get("/")
async def root():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
