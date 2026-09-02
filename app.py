import os
import re
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import TypedDict
from tavily import TavilyClient
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="DeepResearch AI | Smart Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Management ---
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# --- Custom CSS & Animation Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }

    .agent-pulse {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 9999px;
        color: #818cf8;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 12px;
        animation: pulseGlow 2s infinite ease-in-out;
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.3); }
        50% { box-shadow: 0 0 0 8px rgba(99, 102, 241, 0); }
    }

    .verdict-box {
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 0.95rem;
    }
    .verdict-approved {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
    }
    .verdict-rejected {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
    }

    .source-chip {
        display: inline-block;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 4px 4px 4px 0;
        color: #94a3b8;
        text-decoration: none;
        word-break: break-all;
    }
    .source-chip:hover {
        border-color: #6366f1;
        color: #c7d2fe;
    }
</style>
""", unsafe_allow_html=True)

# --- Core Multi-Agent Graph ---
class State(TypedDict):
    topic: str
    web_search_result: str
    urls: list[str]
    scraped_text: dict
    draft: str
    feedback: str
    isApproved: bool
    attempts: int

def get_graph():
    tavily = TavilyClient()
    writer_llm = ChatMistralAI(model="mistral-small-2506", temperature=0.7)
    critic_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    def search_node(state: State) -> dict:
        results = tavily.search(query=state['topic'], max_results=5)
        out, urls = [], []
        for r in results.get('results', []):
            out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n")
            urls.append(r['url'])
        return {"web_search_result": "\n--------\n".join(out), "urls": urls}

    def scraper_node(state: State) -> dict:
        out = {}
        for u in state['urls']:
            try:
                resp = requests.get(u, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                    tag.decompose()
                out[u] = soup.get_text(separator=" ", strip=True)[:3000]
            except Exception as e:
                out[u] = f"Could not access article: {str(e)}"
        return {"scraped_text": out}

    def writer_node(state: State) -> dict:
        attempt = state['attempts'] + 1
        if attempt == 1:
            user_message = (
                f"Write clear, structured, and insightful reports on this topic: {state['topic']}\n"
                f"Research gathered:\n{state['scraped_text']}"
            )
        else:
            user_message = (
                f"Your previous draft on '{state['topic']}' was reviewed with notes.\n"
                f"Reviewer notes:\n{state['feedback']}\n\n"
                f"Previous draft:\n{state['draft']}\n\n"
                f"Produce an improved report resolving every observation raised above."
            )
        
        system_prompt = (
            "You are an expert research report writer. Return ONLY the markdown report following these headers: "
            "# Introduction, # Key Findings, # Analysis, # Conclusion, # Sources. "
            "Never fabricate facts. Always list the URLs used under Sources."
        )
        response = writer_llm.invoke([("system", system_prompt), ("human", user_message)])
        return {"draft": response.content, "attempts": attempt}

    def critic_node(state: State) -> dict:
        prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Research Report Draft:\n{state['draft']}\n\n"
            "Review this draft against strict accuracy, depth, structure, and source integrity criteria.\n"
            "Respond in this EXACT format:\n"
            "SCORE: <number>/10\n"
            "VERDICT: APPROVED or REJECTED\n"
            "FEEDBACK: <constructive explanation>"
        )
        response = critic_llm.invoke([
            ("system", "You are a constructive editorial quality reviewer."),
            ("human", prompt)
        ])
        content = response.content.strip()
        is_approved = "VERDICT: APPROVED" in content
        feedback = content.split("FEEDBACK:", 1)[-1].strip() if "FEEDBACK:" in content else content
        return {"feedback": feedback, "isApproved": is_approved}

    def router(state: State):
        if state['isApproved'] or state['attempts'] >= 3:
            return END
        return "Writer_node"

    builder = StateGraph(State)
    builder.add_node("search_node", search_node)
    builder.add_node("scraper_node", scraper_node)
    builder.add_node("Writer_node", writer_node)
    builder.add_node("Critics_node", critic_node)

    builder.add_edge(START, "search_node")
    builder.add_edge("search_node", "scraper_node")
    builder.add_edge("scraper_node", "Writer_node")
    builder.add_edge("Writer_node", "Critics_node")
    builder.add_conditional_edges("Critics_node", router)

    return builder.compile()

# --- Smooth Scroll Utility ---
def trigger_scroll(target_id: str):
    st.components.v1.html(
        f"""
        <script>
            setTimeout(function() {{
                const target = window.parent.document.getElementById('{target_id}');
                if (target) {{
                    target.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
            }}, 200);
        </script>
        """,
        height=0,
    )

# --- Friendly Sidebar ---
with st.sidebar:
    st.markdown("### 💡 How It Works")
    st.write(
        "This assistant assigns multiple AI specialists to build a verified, well-cited briefing for you:"
    )
    st.markdown("""
    1. **🌐 Web Finder:** Discovers relevant publications and news.
    2. **📖 Content Reader:** Extracts key context and data points.
    3. **✍️ Report Writer:** Synthesizes the information into clear sections.
    4. **🔍 Quality Reviewer:** Fact-checks and refines the draft.
    """)
    st.divider()
    st.caption("Tip: Detailed topics yield more focused insights and statistics.")

# --- Main Interface ---
st.title("🔬 DeepResearch AI")
st.markdown("Enter any subject to generate a factual, double-checked briefing backed by real-world sources.")

# Example prompts for everyday users
def set_topic(example_text):
    st.session_state.topic_input = example_text

st.write("Need inspiration? Try one of these:")
col_ex1, col_ex2, col_ex3 = st.columns(3)
with col_ex1:
    if st.button("🔋 Solid-State Battery Future", use_container_width=True, disabled=st.session_state.is_running):
        set_topic("Recent breakthroughs in solid-state batteries for electric vehicles")
with col_ex2:
    if st.button("🌱 Vertical Farming Economics", use_container_width=True, disabled=st.session_state.is_running):
        set_topic("Economic viability and sustainability of urban vertical farming")
with col_ex3:
    if st.button("🧠 AI in Medical Diagnostics", use_container_width=True, disabled=st.session_state.is_running):
        set_topic("Clinical adoption and accuracy of AI diagnostics in healthcare")

topic_query = st.text_input(
    "What would you like to research?",
    value=st.session_state.topic_input,
    placeholder="Type a subject or question (e.g. Current trends in renewable geothermal energy)...",
    disabled=st.session_state.is_running
)

# Button state: Disabled while running or if the text field is empty
btn_label = "⏳ Researching... Please wait" if st.session_state.is_running else "🚀 Start Research"
start_btn = st.button(
    btn_label,
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_running or len(topic_query.strip()) == 0
)

# Placeholders
status_area = st.empty()
metrics_col = st.empty()
tabs_container = st.empty()

# --- Execution Workflow ---
if start_btn:
    st.session_state.is_running = True
    st.rerun()

if st.session_state.is_running:
    st.markdown("<div id='research_viewport'></div>", unsafe_allow_html=True)
    trigger_scroll("research_viewport")

    app = get_graph()
    initial_state: State = {
        "topic": topic_query,
        "web_search_result": "",
        "urls": [],
        "scraped_text": {},
        "draft": "",
        "feedback": "",
        "isApproved": False,
        "attempts": 0
    }

    current_state = initial_state.copy()

    with status_area.container():
        st.markdown("<div class='agent-pulse'>🚀 Starting research agents...</div>", unsafe_allow_html=True)

    try:
        for output in app.stream(initial_state):
            for node_name, node_update in output.items():
                current_state.update(node_update)

                with status_area.container():
                    if node_name == "search_node":
                        st.markdown(
                            "<div class='agent-pulse'>🔍 Searching the web for reliable sources...</div>",
                            unsafe_allow_html=True
                        )
                    elif node_name == "scraper_node":
                        st.markdown(
                            f"<div class='agent-pulse'>📖 Reading articles ({len(current_state.get('urls', []))} pages found)...</div>",
                            unsafe_allow_html=True
                        )
                    elif node_name == "Writer_node":
                        attempt = current_state.get('attempts', 1)
                        st.markdown(
                            f"<div class='agent-pulse'>✍️ Drafting report (Version #{attempt})...</div>",
                            unsafe_allow_html=True
                        )
                    elif node_name == "Critics_node":
                        st.markdown(
                            "<div class='agent-pulse'>🔍 Reviewing draft for clarity and accuracy...</div>",
                            unsafe_allow_html=True
                        )

        st.session_state.research_data = current_state
    finally:
        st.session_state.is_running = False
        st.rerun()

# --- Display Persisted Results ---
if st.session_state.research_data and not st.session_state.is_running:
    status_area.empty()
    res = st.session_state.research_data

    # Key Metrics
    with metrics_col.container():
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f"<div class='metric-card'><span style='color:#94a3b8;font-size:0.8rem;'>SOURCES READ</span><br>"
                f"<span style='font-size:1.6rem;font-weight:700;'>{len(res.get('urls', []))}</span></div>",
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'><span style='color:#94a3b8;font-size:0.8rem;'>DRAFT REVISIONS</span><br>"
                f"<span style='font-size:1.6rem;font-weight:700;'>{res.get('attempts', 1)}</span></div>",
                unsafe_allow_html=True
            )
        with m3:
            approved = res.get('isApproved', False)
            verdict_markup = (
                "<span style='color:#4ade80;font-size:1.6rem;font-weight:700;'>Verified ✓</span>"
                if approved else
                "<span style='color:#facc15;font-size:1.6rem;font-weight:700;'>Draft Ready</span>"
            )
            st.markdown(
                f"<div class='metric-card'><span style='color:#94a3b8;font-size:0.8rem;'>QUALITY STATUS</span><br>"
                f"{verdict_markup}</div>",
                unsafe_allow_html=True
            )
        with m4:
            word_count = len(res.get('draft', '').split())
            st.markdown(
                f"<div class='metric-card'><span style='color:#94a3b8;font-size:0.8rem;'>REPORT LENGTH</span><br>"
                f"<span style='font-size:1.6rem;font-weight:700;'>{word_count} words</span></div>",
                unsafe_allow_html=True
            )
        st.write("")

    # Result Tabs
    with tabs_container.container():
        tab_report, tab_sources, tab_review = st.tabs([
            "📑 Research Report", 
            "🔗 Sources Used", 
            "📝 Quality Review Notes"
        ])

        with tab_report:
            st.markdown(res.get("draft", ""))
            st.download_button(
                label="📥 Download Report (.md)",
                data=res.get("draft", ""),
                file_name=f"research_{re.sub(r'[^a-zA-Z0-9]', '_', topic_query)[:20]}.md",
                mime="text/markdown",
                use_container_width=True
            )

        with tab_sources:
            st.subheader("Referenced Web Sources")
            urls = res.get("urls", [])
            if urls:
                for url in urls:
                    st.markdown(f"<a class='source-chip' href='{url}' target='_blank'>🔗 {url}</a>", unsafe_allow_html=True)
            else:
                st.write("No external URLs were returned.")

            st.divider()
            st.subheader("Extracted Reading Highlights")
            for url, raw in res.get("scraped_text", {}).items():
                with st.expander(f"📄 Highlights from: {url}", expanded=False):
                    st.text(raw[:1500] + ("..." if len(raw) > 1500 else ""))

        with tab_review:
            st.subheader("Reviewer Assessment")
            if res.get("isApproved", False):
                st.markdown(
                    "<div class='verdict-box verdict-approved'><strong>✅ Quality Standard Met</strong><br>"
                    "The report passed editorial fact-checks and accurately synthesizes the gathered sources.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='verdict-box verdict-rejected'><strong>ℹ️ Informational Draft</strong><br>"
                    "The report completed all designated refinement cycles and is presented with the latest notes below.</div>",
                    unsafe_allow_html=True
                )

            st.markdown("#### Reviewer Feedback")
            st.info(res.get("feedback", "No specific feedback was noted."))