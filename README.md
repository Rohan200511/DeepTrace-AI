# DeepTrace AI

DeepTrace AI is a multi-agent research assistant that finds, reads, and synthesizes high-quality, cited research briefings. It combines web search (Tavily), content scraping, and multiple LLM roles (writer + critic) to produce structured, verifiable research reports. A Streamlit UI is provided for interactive use, and a CLI flow is available for offline runs.

Key features

- Multi-agent pipeline: search → scrape → draft → critique → revise
- Structured output with sections: Introduction, Key Findings, Analysis, Conclusion, Sources
- Source-tracing: all reports include URLs used during research
- Streamlit-based web UI for interactive research sessions
- Configurable LLM backends (Mistral, Google Generative AI)

Quick demo

1. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Create a .env file with the following environment variables (examples):

   ```text
   TAVILY_API_KEY=your_tavily_api_key
   # Depending on LLM backends you use, also provide credentials/keys for those providers
   # For Mistral or LangChain integrations, follow their SDK instructions
   GOOGLE_API_KEY=your_google_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ```

3. Run the Streamlit app (recommended for most users):

   ```bash
   streamlit run app.py
   ```

4. Or run the CLI prototype:

   ```bash
   python main.py
   ```

Project layout

- app.py — Streamlit application: provides the interactive UI, session management, and runs the multi-agent StateGraph flow.
- main.py — CLI prototype for the multi-agent pipeline (search, scrape, write, criticize).
- DeepTrace_Documentation.pdf — project documentation (included in repo).
- requirements.txt — Python dependencies.

Architecture overview

DeepTrace uses a small state graph that wires together specialized nodes:

- search_node: uses Tavily to find relevant webpages for a given topic
- scraper_node: downloads and extracts plain text from found URLs (BeautifulSoup + requests)
- Writer_node: a dedicated LLM role (Mistral) that drafts a structured research report
- Critics_node: a stricter LLM role (Google Generative) that reviews the draft and returns a verdict + feedback

The router repeatedly sends the draft back to the writer for revision until the critic approves or a maximum attempt count is reached.

Environment & configuration

- Ensure keys for Tavily and any LLM provider are present in your environment or .env file.
- The project relies on these packages (see requirements.txt). Some provider SDKs may need additional configuration (authentication, network access).

Security and usage notes

- The app makes outbound HTTP requests to fetch source pages. Use with care regarding rate limits and target site scraping policies (robots.txt).
- Avoid exposing your API keys in public forks — keep them in a local .env or secret manager.

Extending the project

- Swap or add LLM backends by replacing/injecting the ChatMistralAI and ChatGoogleGenerativeAI clients with other LangChain-compatible classes.
- Add more robust scraping (pagination, article extraction libraries like newspaper3k) or caching for repeatability.
- Add unit tests around scraper parsing and router logic.

Contributing

Contributions are welcome. File issues describing bugs or feature requests, and submit PRs with clear descriptions and tests where possible.

License

This repository does not specify a license. Consider adding an explicit license (e.g., MIT) if you want to grant reuse rights.

Acknowledgements

- Built with Tavily for search, Mistral & Google generative models for drafting/review, Streamlit for UI, and Requests + BeautifulSoup for scraping.

Contact

For questions or help, open an issue or contact the repository owner.
