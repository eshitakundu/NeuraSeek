from typing import List, Tuple
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.prompts import ChatPromptTemplate


class RAGChain:    
    def __init__(self, api_key: str, model: str = "meta/llama-3.1-8b-instruct"):
        self.llm = ChatNVIDIA(
            model=model,
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Answer the user's question based on the provided context.
If the answer cannot be found in the context, say so politely and provide general information if helpful.

Context:
{context}"""),
            ("user", "{question}")
        ])
    
    def generate_response(
        self, 
        question: str, 
        context_chunks: List[Tuple[str, dict]]
    ) -> str:
        if not context_chunks:
            return "I don't have any documents to reference. Please upload a document first."
        
        # Combine context
        context = "\n\n".join([chunk for chunk, _ in context_chunks])
        
        # Generate prompt
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        # Get response
        response = self.llm.invoke(messages)
        return response.content