# agent.py
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Type, Dict, Any, Generator
from pydantic import BaseModel, Field
from config import INDEX_A, INDEX_B, SENTENCE_MODEL, OLLAMA_MODEL
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set dummy OpenAI API key (CrewAI still checks for it)
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-llm"

# ==================== GLOBAL RETRIEVERS ====================

embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{SENTENCE_MODEL}")
db_a = FAISS.load_local(str(INDEX_A), embeddings, allow_dangerous_deserialization=True)
db_b = FAISS.load_local(str(INDEX_B), embeddings, allow_dangerous_deserialization=True)

retriever_a = db_a.as_retriever(search_kwargs={"k": 4})
retriever_b = db_b.as_retriever(search_kwargs={"k": 4})


# ==================== CUSTOM TOOLS ====================

class FAISSSearchToolInput(BaseModel):
    """Input schema for FAISS search tool"""
    query: str = Field(..., description="The search query to look up in the vector database")
    company: str = Field(..., description="Company to search: 'A', 'B', or 'BOTH'")


class FAISSSearchTool(BaseTool):
    """Custom tool for searching FAISS vector databases"""
    name: str = "faiss_search"
    description: str = (
        "Search for information in Company A or Company B financial documents. "
        "Use this tool to retrieve relevant context from vector databases. "
        "Specify 'A' for Company A, 'B' for Company B, or 'BOTH' for both companies."
    )
    args_schema: Type[BaseModel] = FAISSSearchToolInput
    
    def _run(self, query: str, company: str) -> str:
        """Execute the search using global retrievers"""
        logger.info(f"FAISS Search Tool called: query='{query}', company='{company}'")
        docs = []
        company = company.upper()
        
        if company in ("A", "BOTH"):
            docs += retriever_a.invoke(query)
        if company in ("B", "BOTH"):
            docs += retriever_b.invoke(query)
        
        formatted = []
        for d in docs:
            comp = d.metadata.get("company", "Unknown")
            formatted.append(f"[{comp}]\n{d.page_content}")
        
        result = "\n\n".join(formatted) if formatted else "No relevant information found."
        return result


# ==================== LLM INITIALIZATION ====================

# For CrewAI agents
llm_string = f'ollama/{OLLAMA_MODEL}'

# For streaming (direct LangChain access)
streaming_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url="http://localhost:11434",
    temperature=0.2,
    streaming=True  # Enable token streaming
)


# ==================== STREAMING FUNCTIONS ====================

def answer_question_streaming(question: str) -> Generator[Dict[str, Any], None, None]:
    """
    True token-by-token streaming with multi-step agentic workflow.
    Yields dictionaries with 'type' and 'content' keys for UI rendering.
    
    Yields:
        - {'type': 'status', 'content': 'message'} for status updates
        - {'type': 'step', 'content': 'step_name'} for workflow steps
        - {'type': 'token', 'content': 'text'} for streaming tokens
        - {'type': 'context', 'content': {...}} for retrieved context
    """
    logger.info(f"Starting streaming workflow for: {question}")
    
    # ===== STEP 1: QUERY DECOMPOSITION =====
    yield {'type': 'status', 'content': '🔍 Analyzing and decomposing your question...'}
    yield {'type': 'step', 'content': 'decomposition'}
    
    decompose_prompt = ChatPromptTemplate.from_template("""
You are a query decomposition expert. Break down this question into 1-3 simple sub-questions.

Rules:
- Tag each with 'A:' if it's about Company A only
- Tag each with 'B:' if it's about Company B only  
- Tag each with 'BOTH:' if it requires both companies
- Keep sub-questions simple and focused

Question: {question}

Sub-questions (one per line):""")
    
    decompose_chain = decompose_prompt | streaming_llm | StrOutputParser()
    
    sub_questions_text = ""
    for chunk in decompose_chain.stream({"question": question}):
        sub_questions_text += chunk
        yield {'type': 'token', 'content': chunk}
    
    yield {'type': 'token', 'content': '\n\n'}
    
    # Parse sub-questions
    sub_questions = [q.strip() for q in sub_questions_text.strip().split('\n') if q.strip()]
    logger.info(f"Decomposed into {len(sub_questions)} sub-questions: {sub_questions}")
    
    # ===== STEP 2: VECTOR DATABASE RESEARCH =====
    yield {'type': 'status', 'content': f'📚 Searching {len(sub_questions)} queries across Company A and B databases...'}
    yield {'type': 'step', 'content': 'research'}
    
    all_contexts = []
    research_results = []
    
    for i, sq in enumerate(sub_questions, 1):
        # Determine routing
        if sq.startswith('A:'):
            company = 'A'
            query_text = sq[2:].strip()
        elif sq.startswith('B:'):
            company = 'B'
            query_text = sq[2:].strip()
        elif sq.startswith('BOTH:'):
            company = 'BOTH'
            query_text = sq[5:].strip()
        else:
            company = 'BOTH'
            query_text = sq
        
        # Retrieve documents
        docs = []
        if company in ('A', 'BOTH'):
            docs_a = retriever_a.invoke(query_text)
            for d in docs_a:
                all_contexts.append(f"[Company A]\n{d.page_content}")
                docs.append({'company': 'A', 'content': d.page_content})
        
        if company in ('B', 'BOTH'):
            docs_b = retriever_b.invoke(query_text)
            for d in docs_b:
                all_contexts.append(f"[Company B]\n{d.page_content}")
                docs.append({'company': 'B', 'content': d.page_content})
        
        research_results.append({
            'sub_question': sq,
            'query': query_text,
            'company': company,
            'docs_found': len(docs)
        })
        
        yield {'type': 'token', 'content': f"**Sub-query {i}:** {query_text} → Found {len(docs)} documents from Company {company}\n\n"}
    
    # Yield context info
    yield {'type': 'context', 'content': {
        'total_documents': len(all_contexts),
        'research_results': research_results
    }}
    
    context_text = "\n\n".join(all_contexts[:10])  # Limit to top 10 most relevant
    logger.info(f"Retrieved total {len(all_contexts)} context chunks")
    
    # ===== STEP 3: ANSWER SYNTHESIS =====
    yield {'type': 'status', 'content': '🧠 Synthesizing comprehensive answer from research findings...'}
    yield {'type': 'step', 'content': 'synthesis'}
    yield {'type': 'token', 'content': '\n\n---\n\n## 💡 Final Answer\n\n'}
    
    answer_prompt = ChatPromptTemplate.from_template("""
You are a financial analyst synthesizing information from multiple sources.

Context from Company A and Company B documents:
{context}

Original Question: {question}

Instructions:
1. Provide a clear, comprehensive answer
2. Use specific numbers, dates, and facts from the context
3. If comparing companies, present comparisons clearly
4. Format with bullet points or paragraphs as appropriate
5. Be concise but complete

Answer:""")
    
    answer_chain = answer_prompt | streaming_llm | StrOutputParser()
    
    for chunk in answer_chain.stream({"context": context_text, "question": question}):
        yield {'type': 'token', 'content': chunk}
    
    yield {'type': 'status', 'content': '✅ Analysis complete!'}
    logger.info("Streaming workflow completed successfully")


def answer_question_crewai(question: str, verbose: bool = True) -> str:
    """
    Non-streaming CrewAI version for comparison or fallback.
    Uses the full multi-agent system.
    """
    logger.info(f"Processing question with CrewAI: {question}")
    
    search_tool = FAISSSearchTool()
    
    # Create agents
    decomposer = Agent(
        role="Query Decomposition Specialist",
        goal="Break down complex multi-part questions into simple, independent sub-questions",
        backstory="Expert at analyzing complex questions and breaking them down.",
        llm=llm_string,
        verbose=True,
        allow_delegation=False,
        max_iter=15
    )
    
    researcher = Agent(
        role="Financial Research Analyst",
        goal="Retrieve accurate financial information from company documents",
        backstory="Meticulous analyst who retrieves precise information from databases.",
        llm=llm_string,
        tools=[search_tool],
        verbose=True,
        allow_delegation=False,
        max_iter=20
    )
    
    synthesizer = Agent(
        role="Answer Synthesis Specialist",
        goal="Combine multiple research findings into clear answers",
        backstory="Expert at synthesizing information into coherent answers.",
        llm=llm_string,
        verbose=True,
        allow_delegation=False,
        max_iter=15
    )
    
    # Create tasks
    decomposition_task = Task(
        description=f"Break down this question: {question}\nFormat: A:, B:, or BOTH: prefix per line",
        expected_output="List of sub-questions with prefixes",
        agent=decomposer
    )
    
    research_task = Task(
        description="Research each sub-question using faiss_search tool. Provide Q&A pairs.",
        expected_output="Q&A pairs for each sub-question",
        agent=researcher,
        context=[decomposition_task]
    )
    
    synthesis_task = Task(
        description=f"Synthesize findings into final answer for: {question}",
        expected_output="Clear, comprehensive answer",
        agent=synthesizer,
        context=[research_task]
    )
    
    # Execute crew
    crew = Crew(
        agents=[decomposer, researcher, synthesizer],
        tasks=[decomposition_task, research_task, synthesis_task],
        process=Process.sequential,
        verbose=verbose,
        memory=False,
        cache=False
    )
    
    result = crew.kickoff()
    return result.raw if hasattr(result, 'raw') else str(result)


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_answer(question: str) -> str:
    """Quick non-streaming answer using CrewAI"""
    return answer_question_crewai(question, verbose=False)


if __name__ == "__main__":
    # Test streaming
    print("Testing streaming workflow...\n")
    
    test_question = "What are the revenues of Company A and Company B?"
    
    for event in answer_question_streaming(test_question):
        if event['type'] == 'token':
            print(event['content'], end='', flush=True)
        elif event['type'] == 'status':
            print(f"\n[STATUS] {event['content']}\n")
        elif event['type'] == 'step':
            print(f"\n--- {event['content'].upper()} ---\n")
