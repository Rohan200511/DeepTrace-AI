import os
import requests
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict , Annotated
from langgraph.graph import StateGraph , START , END
from langchain_mistralai import ChatMistralAI  #Write
from langchain_google_genai import ChatGoogleGenerativeAI #Criticize
from tavily import TavilyClient
from langgraph.graph.message import add_messages 
from bs4 import BeautifulSoup


class State(TypedDict):
    topic : str
    web_search_result : str #URL , Title , content
    urls : list[str]
    scraped_text : list[dict]
    draft : str
    feedback : str
    isApproved : bool
    attempts : int

tavily = TavilyClient()

def search(state : State) -> dict:
    """Search the topic using TavilySearch Tool"""
    topic = state['topic']
    
    results = tavily.search(query=topic , max_results=5)
    out = []
    urls = []
    for r in results['results']:
            out.append(
                f"Title: {r['title']}\n URL: {r['url']}\n Snippet: {r['content'][:300]}\n"
            )
            urls.append(r['url'])
            
    return {"web_search_result" : "\n--------\n".join(out),
             "urls" : urls
            }
        

def scraper(state : State) -> dict:
    """Scrape and return clean text content from the URLs for deeper reading."""
    urls = state['urls']
    
    out = {}
    
    for u in urls:
        try:
            resp = requests.get(u , timeout=8 , headers={"User-Agent" : "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text , "html.parser")
                    
            for tag in soup(['script' , 'style' , 'nav' , 'footer']):
                tag.decompose()
            
            out[u] = soup.get_text(separator=" " , strip=True)[:3000]
        
        except Exception as e:
            out[u] = f"Could not scrape URL: {str(e)}"
    
    return{
        "scraped_text" : out
    }
    
writer_llm =ChatMistralAI(model = "mistral-small-2506" , temperature=0.7)

WRITER_SYSTEM_PROMPT = (
    """You are an expert research report writer.
    
    Your task is to transform the provided research material into a clear, accurate, well-structured, and professional research report.

    Follow these rules:

    1. Use ONLY the information provided in the research material.
    2. Do not invent facts, statistics, examples, or sources.
    3. If the available research is insufficient to support a claim, do not make that claim.
    4. Synthesize information from multiple sources instead of simply copying individual sources.
    5. Keep the writing objective, factual, and easy to understand.
    6. Clearly distinguish important findings, supporting evidence, and conclusions.
    7. Avoid unnecessary repetition and filler content.
    8. Preserve important source URLs so that claims can be traced back to their sources.
    9. If critic feedback is provided, revise the previous draft according to that feedback while preserving correct and useful content.
    10. Return ONLY the research report, without explaining your writing process.

    Use the following structure:

    # Introduction
    Briefly introduce the topic and explain its significance.

    # Key Findings
    Present at least 3 important findings.
    For each finding:
    - State the main point clearly.
    - Explain it using the available evidence.
    - Mention the relevant source when appropriate.

    # Analysis
    Synthesize the findings and explain the broader implications, relationships, or patterns revealed by the research.

    # Conclusion
    Summarize the most important insights and provide a concise concluding perspective.

    # Sources
    List all sources used in the report with their URLs."""
)

def writer_node(state:State) -> dict:
    """Writes the Content on the basis of Research Gathered in scrapped_text"""
    topic = state['topic']
    research = state['scraped_text']
    attempt = state['attempts'] + 1
    prev_feedback = state['feedback']
    
    if attempt == 1:
            user_message = (
                f"Write clear, structured and insightful reports on this topic {topic}"
                f"Some research is gathered already as {research} "
            )
    else:
        user_message = (
            f"your previous draft on '{topic}' was rejected"
            f"Here is the reviewer's feedback \n\n {prev_feedback}\n\n"
            f"Here is yout previous draft {state['draft']}"
            f"Write a new, improved draft that fixes every issue mentiond"
            f"do not repeat the same mistake"
        )
    
    messages = [("system" , WRITER_SYSTEM_PROMPT) , ("human" , user_message)]
    
    response = writer_llm.invoke(messages)
    
    return{
        "draft" : response.content,
        "attempts" : attempt
    }
    
critic_llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , temperature = 0.1)

CRITICS_SYSTEM_PROMPT = (
    "You are a strict and constructive research report reviewer. "
    "Your job is to evaluate whether a research report is accurate, "
    "well-structured, well-supported, and ready for final delivery.\n\n"

    "Evaluate the report against these criteria:\n"
    "1. Accuracy — claims must be supported by the provided research.\n"
    "2. Research coverage — important information from the gathered research "
    "should be reflected in the report.\n"
    "3. Structure — clear introduction, key findings, analysis, conclusion, "
    "and sources.\n"
    "4. Depth — findings should be explained rather than merely listed.\n"
    "5. Relevance — the report should stay focused on the given topic.\n"
    "6. Clarity — writing should be clear, concise, professional, and easy to read.\n"
    "7. Sources — relevant source URLs should be included and claims should "
    "be traceable to the provided sources.\n"
    "8. No unsupported or fabricated facts, statistics, or sources.\n\n"

    "Respond in exactly this format:\n"
    "SCORE: <integer>/10\n"
    "VERDICT: APPROVED or REJECTED\n"
    "FEEDBACK: <specific explanation of what is good and what needs improvement>\n\n"

    "Be strict but fair. Approve only if the report meets the criteria "
    "well enough to be considered a high-quality final research report. "
    "If important issues remain, reject it and clearly explain what the "
    "writer must improve."
)

def critic_node(state: State) -> dict:
    """Reviews the draft and decides: approve or reject with feedback."""
    
    topic = state["topic"]
    draft = state["draft"]

    prompt = (
        f"Topic: {topic}\n\n"
        f"Research Report Draft:\n{draft}\n\n"
        "Review this research report and give your verdict and feedback."
    )

    messages = [
        ("system", CRITICS_SYSTEM_PROMPT),
        ("human", prompt)
    ]

    response = critic_llm.invoke(messages)
    review_text = response.content.strip()

    is_approved = "VERDICT: APPROVED" in review_text

    feedback = review_text.split("FEEDBACK:", 1)[-1].strip()

    return {
        "feedback": feedback,
        "isApproved": is_approved
    }
    
def router(state:State):
    is_approved = state['isApproved']
    attempts = state['attempts']
    
    if is_approved: 
        return END
    
    elif attempts < 3:
        return "Writer_node"
    
    else:
        return END
    
    

graph = StateGraph(State)

graph.add_node("search_node", search)
graph.add_node("scrapper_node", scraper)
graph.add_node("Writer_node", writer_node)
graph.add_node("Critics_node", critic_node)

graph.add_edge(START, "search_node")
graph.add_edge("search_node", "scrapper_node")
graph.add_edge("scrapper_node", "Writer_node")
graph.add_edge("Writer_node", "Critics_node")

graph.add_conditional_edges("Critics_node", router)

app = graph.compile()

print("=" * 40)
print("Welcome to Multi Agent Research System")
print("=" * 40)

topic = input("Enter the topic you want research on: ")

print()
print()

initial_state = {
        "topic" : topic,
        "web_search_result" : "" ,
        "urls" : [],
        "scraped_text" : [],
        "draft" : "",
        "feedback" : "",
        "isApproved" : False,
        "attempts" : 0
    }

final_state = app.invoke(initial_state)

print()
print("Final Research with Feedback: ")
print()
print()
print()
print("Research: ")
print("--" * 20)
print(final_state['draft'])

print("Feedback: ")
print("--" * 20)
print(final_state["feedback"])

print()
print("Total Attempts:" , final_state["attempts"])