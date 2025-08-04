import os
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document  # Required for handling documents

# Global variables to hold the language model and document chunks
llm = None
all_texts: List[Document] = []
prompt = None

class QuestionRequest(BaseModel):
    question: str

def initialize_chatbot():
    """
    Initializes the language model, loads and processes the PDF,
    and prepares the prompt template.
    """
    global llm, all_texts, prompt

    # --- LLM and API Key Setup ---
    # Ensure your GOOGLE_API_KEY is set as a secret in your deployment environment
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, max_tokens=300)

    # --- Document Loading and Splitting (replaces FAISS) ---
    # This part runs once on startup
    try:
        loader = PyPDFLoader("teja.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        all_texts = text_splitter.split_documents(documents)
    except Exception as e:
        # If the PDF can't be loaded, store an error message as the context.
        error_message = f"Error loading or processing PDF: {e}"
        all_texts = [Document(page_content=error_message)]
        print(error_message)


    # --- Prompt Template ---
    # This defines how we structure the query to the LLM
    prompt = PromptTemplate(
        template="""You are answering questions about Suryateja's profile. Use the provided context to answer the question accurately and comprehensively.

INSTRUCTIONS:
- Search through ALL the provided context carefully.
- Look for direct mentions, related terms, and implicit references.
- If you find relevant information, provide a complete answer.
- If the exact information isn't available but related information exists, mention what is available.
- Only say "Information not available" if there's truly no relevant information in the context.

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

def generate_questions(context: str, current_answer: str) -> List[str]:
    """
    Generates follow-up questions based on the conversation.
    """
    if not llm:
        return []

    try:
        response = llm.invoke([{"role": "user", "content":
            f"""Based on the following context about Suryateja and the current answer provided, generate 3 specific, factual follow-up questions that are related to the current answer and can be answered from the context. The questions should naturally flow from the current conversation.

Context: {context[:1500]}

Current Answer: {current_answer}

Generate 3 related follow-up questions (each ending with '?'):
Note:
Ensure the questions are specific and can be answered with the provided context.
If there is no change of question related to the current answer, then generate questions based on the context.
The questions will be answered by the chatbot, not the user.
"""
        }])

        questions = [line.strip() for line in response.content.split('\n')
                    if '?' in line and len(line.strip()) > 10][:3]
    except Exception:
        questions = [] # Return empty list if generation fails

    # Fallback questions if generation fails or returns no questions
    if not questions:
        questions = [
            "What programming languages does Suryateja know?",
            "Which university is Suryateja studying at?",
            "What is Suryateja's current CGPA?"
        ]

    return questions

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Initializing chatbot...")
    initialize_chatbot()
    print("Chatbot initialized.")
    yield
    # This code runs on shutdown (if needed)
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: QuestionRequest):
    """
    Handles the chat request by combining documents and querying the LLM.
    """
    # Create an LLMChain and a StuffDocumentsChain to process the request
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Invoke the chain with all document chunks
    result = stuff_chain.invoke({
        "input_documents": all_texts,
        "question": request.question
    })
    answer = result["output_text"]

    # Generate follow-up questions
    context_for_followup = " ".join([doc.page_content for doc in all_texts[:5]]) # Use first few chunks for context
    questions = generate_questions(context_for_followup, answer)

    return {"result": [answer] + questions}

@app.get("/")
async def root():
    return {"status": "running"}

# This block is for local testing and will not be used by most deployment services
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
