"""
Production-ready Financial Intelligence Dashboard
=================================================

Run:
    pip install -r requirements.txt
    streamlit run financial_intelligence_dashboard.py

What this app does:
- Upload and index a financial PDF with LlamaIndex + ChromaDB.
- Uses advanced sentence-window document parsing to preserve local context.
- Creates a ReAct agent with two tools:
    1. financial_report_pdf: RAG over the uploaded PDF.
    2. market_data_search: yfinance quote, valuation, history, and news lookup.
- Displays a safe action trace, citations/source chunks, and a live-ish price chart.

Note: yfinance uses Yahoo Finance data and may be delayed or rate-limited. Do not use
this app as financial advice or as a direct trading system without independent checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import chromadb
import pandas as pd
import streamlit as st
import yfinance as yf
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.agent.workflow import AgentStream, ReActAgent, ToolCallResult
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


# -----------------------------------------------------------------------------
# Constants and page setup
# -----------------------------------------------------------------------------

APP_TITLE = "Financial Intelligence Dashboard"
APP_SUBTITLE = "Agentic RAG over filings + live market intelligence"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

for folder in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """
You are a senior financial intelligence analyst.

You have two tools:
1. financial_report_pdf: use this for facts, figures, disclosures, risks, and text from the uploaded PDF.
2. market_data_search: use this for stock price, valuation ratio, recent price history, and news.

Rules:
- Use the uploaded PDF for document-specific claims.
- Use market_data_search for ticker/market claims.
- If the user asks for comparison between company fundamentals and market data, use both tools.
- Never invent missing financial metrics. Say when a metric is unavailable.
- Be concise, analytical, and explicit about assumptions.
- Do not provide investment advice; frame outputs as analysis, not a recommendation to buy/sell.
""".strip()


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class SidebarConfig(BaseModel):
    """Validated runtime settings from the sidebar."""

    openai_api_key: SecretStr
    ticker: str = Field(default="AAPL", min_length=1, max_length=12)
    llm_model: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    parser_strategy: Literal["sentence_window", "semantic_splitter"] = "sentence_window"
    similarity_top_k: int = Field(default=5, ge=1, le=12)
    chart_period: Literal["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"] = "6mo"
    chart_interval: Literal["1d", "1wk", "1mo"] = "1d"
    reset_index: bool = False

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
            raise ValueError("Ticker must contain only letters, numbers, dots, or hyphens.")
        return ticker


class StockNewsItem(BaseModel):
    title: str
    publisher: Optional[str] = None
    link: Optional[str] = None
    published: Optional[str] = None


class StockSnapshot(BaseModel):
    ticker: str
    as_of_utc: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    pe_ratio: Optional[float] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    news: List[StockNewsItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class SourceCitation(BaseModel):
    citation_id: str
    file_name: Optional[str] = None
    page_label: Optional[str] = None
    score: Optional[float] = None
    text: str


class AgentRunResult(BaseModel):
    answer: str
    action_trace: List[str] = Field(default_factory=list)
    sources: List[SourceCitation] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],
        "indexed_pdf_hash": None,
        "indexed_pdf_name": None,
        "last_sources": [],
        "last_action_trace": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clean_collection_name(raw: str) -> str:
    """Chroma collection names must be 3-63 chars and mostly alphanumeric."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)[:63]
    if len(cleaned) < 3:
        cleaned = f"doc_{cleaned}"
    return cleaned


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_uploaded_pdf(uploaded_file: Any, pdf_hash: str) -> Path:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", uploaded_file.name)
    path = UPLOAD_DIR / f"{pdf_hash[:12]}_{safe_name}"
    path.write_bytes(uploaded_file.getvalue())
    return path


def truncate_text(text: str, max_chars: int = 1_500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def run_coroutine_sync(coro: Any) -> Any:
    """Run an async coroutine from Streamlit's synchronous execution context."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not running_loop.is_running():
        return running_loop.run_until_complete(coro)

    result_box: Dict[str, Any] = {}
    error_box: Dict[str, BaseException] = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["result"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised below with original type
            error_box["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_box:
        raise error_box["error"]
    return result_box.get("result")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_get(mapping_or_obj: Any, key: str, default: Any = None) -> Any:
    try:
        if hasattr(mapping_or_obj, "get"):
            return mapping_or_obj.get(key, default)
        return getattr(mapping_or_obj, key, default)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Market data functions
# -----------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_snapshot_dict(ticker: str) -> Dict[str, Any]:
    """Fetch quote, valuation, and news from yfinance with resilient fallbacks."""
    ticker = ticker.strip().upper()
    warnings: List[str] = []
    stock = yf.Ticker(ticker)

    price: Optional[float] = None
    previous_close: Optional[float] = None
    currency: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

    try:
        fast_info = stock.fast_info
        price = safe_float(safe_get(fast_info, "last_price")) or safe_float(safe_get(fast_info, "lastPrice"))
        previous_close = safe_float(safe_get(fast_info, "previous_close")) or safe_float(
            safe_get(fast_info, "previousClose")
        )
        currency = safe_get(fast_info, "currency")
        market_cap = safe_float(safe_get(fast_info, "market_cap")) or safe_float(safe_get(fast_info, "marketCap"))
    except Exception as exc:
        warnings.append(f"fast_info unavailable: {exc}")

    try:
        hist = stock.history(period="5d", interval="1d", auto_adjust=False)
        if not hist.empty and "Close" in hist:
            close_values = hist["Close"].dropna()
            if not close_values.empty:
                price = safe_float(close_values.iloc[-1]) or price
    except Exception as exc:
        warnings.append(f"price history fallback unavailable: {exc}")

    try:
        info = stock.get_info()
        pe_ratio = safe_float(info.get("trailingPE")) or safe_float(info.get("forwardPE"))
        previous_close = previous_close or safe_float(info.get("previousClose"))
        currency = currency or info.get("currency")
        market_cap = market_cap or safe_float(info.get("marketCap"))
    except Exception as exc:
        warnings.append(f"valuation info unavailable: {exc}")

    parsed_news: List[StockNewsItem] = []
    try:
        raw_news = stock.get_news(count=5) if hasattr(stock, "get_news") else stock.news
        for item in raw_news or []:
            content = item.get("content", item) if isinstance(item, dict) else {}
            title = content.get("title") or item.get("title", "Untitled")
            publisher = content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
            canonical_url = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
            clickthrough_url = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
            link = canonical_url.get("url") or clickthrough_url.get("url") or item.get("link")
            published = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(published, (int, float)):
                published = datetime.fromtimestamp(published, tz=timezone.utc).isoformat()
            parsed_news.append(
                StockNewsItem(
                    title=str(title),
                    publisher=publisher,
                    link=link,
                    published=str(published) if published else None,
                )
            )
    except Exception as exc:
        warnings.append(f"news unavailable: {exc}")

    snapshot = StockSnapshot(
        ticker=ticker,
        as_of_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        price=price,
        previous_close=previous_close,
        pe_ratio=pe_ratio,
        currency=currency,
        market_cap=market_cap,
        news=parsed_news,
        warnings=warnings,
    )
    return snapshot.model_dump()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval, auto_adjust=False)
    if hist.empty:
        return pd.DataFrame(columns=["Close"])
    return hist[["Close"]].dropna()


def market_data_search(
    ticker: Annotated[str, "Stock ticker symbol, for example AAPL, MSFT, NVDA, TSLA, or BRK-B."],
) -> str:
    """Fetch stock price, P/E ratio, market cap, recent price data availability, and recent news for a ticker."""
    try:
        snapshot = fetch_stock_snapshot_dict(ticker)
        return json.dumps(snapshot, indent=2)
    except Exception as exc:
        return json.dumps(
            {
                "ticker": ticker,
                "error": f"Could not fetch yfinance data: {exc}",
                "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )


# -----------------------------------------------------------------------------
# LlamaIndex RAG setup
# -----------------------------------------------------------------------------

def configure_llamaindex(api_key: str, llm_model: str, embedding_model: str) -> Tuple[OpenAI, OpenAIEmbedding]:
    os.environ["OPENAI_API_KEY"] = api_key
    llm = OpenAI(model=llm_model, temperature=0.1, api_key=api_key)
    embed_model = OpenAIEmbedding(model=embedding_model, api_key=api_key)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm, embed_model


def load_pdf_documents(pdf_path: Path) -> Sequence[Any]:
    try:
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    except Exception as exc:
        raise RuntimeError(f"Could not read PDF. Try a text-based PDF instead of a scanned image PDF. Details: {exc}") from exc
    if not docs:
        raise RuntimeError("The PDF loaded successfully, but no readable text was extracted.")
    return docs


def build_nodes(
    docs: Sequence[Any],
    parser_strategy: Literal["sentence_window", "semantic_splitter"],
    embed_model: OpenAIEmbedding,
) -> List[Any]:
    if parser_strategy == "semantic_splitter":
        parser = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        return list(parser.get_nodes_from_documents(docs))

    parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    return list(parser.get_nodes_from_documents(docs))


def build_query_engine(
    pdf_path: Path,
    pdf_hash: str,
    config: SidebarConfig,
) -> Tuple[Any, int, str]:
    """Create or attach to a Chroma-backed VectorStoreIndex and return a query engine."""
    api_key = config.openai_api_key.get_secret_value()
    _llm, embed_model = configure_llamaindex(api_key, config.llm_model, config.embedding_model)

    collection_name = clean_collection_name(f"fin_pdf_{pdf_hash[:24]}")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if config.reset_index:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception:
            pass

    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if chroma_collection.count() > 0 and not config.reset_index:
        try:
            index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        except TypeError:
            index = VectorStoreIndex.from_vector_store(vector_store)
        node_count = int(chroma_collection.count())
    else:
        docs = load_pdf_documents(pdf_path)
        nodes = build_nodes(docs, config.parser_strategy, embed_model)
        if not nodes:
            raise RuntimeError("The parser produced zero nodes. The PDF may be unreadable or empty.")
        index = VectorStoreIndex(nodes, storage_context=storage_context, show_progress=False)
        node_count = len(nodes)

    postprocessors = []
    if config.parser_strategy == "sentence_window":
        postprocessors.append(MetadataReplacementPostProcessor(target_metadata_key="window"))

    query_engine = index.as_query_engine(
        similarity_top_k=config.similarity_top_k,
        node_postprocessors=postprocessors,
        response_mode="compact",
    )
    return query_engine, node_count, collection_name


def create_agent(query_engine: Any, llm: OpenAI) -> ReActAgent:
    document_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="financial_report_pdf",
        description=(
            "Searches and answers questions from the uploaded financial PDF, such as annual reports, "
            "10-Ks, shareholder letters, risk factors, revenue, segment performance, cash flow, "
            "management discussion, and footnotes. Use a detailed natural-language question as input."
        ),
    )
    market_tool = FunctionTool.from_defaults(
        fn=market_data_search,
        name="market_data_search",
        description=(
            "Fetches current/latest available market data using yfinance: stock price, previous close, "
            "P/E ratio, market cap, currency, and recent news for a ticker symbol."
        ),
    )
    return ReActAgent(
        tools=[document_tool, market_tool],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )


# -----------------------------------------------------------------------------
# Agent execution and citations
# -----------------------------------------------------------------------------

def extract_source_citations(source_nodes: Iterable[Any], max_sources: int = 6) -> List[SourceCitation]:
    citations: List[SourceCitation] = []
    for idx, source_node in enumerate(list(source_nodes)[:max_sources], start=1):
        try:
            node = source_node.node
            metadata = getattr(node, "metadata", {}) or {}
            text = metadata.get("window") or metadata.get("original_text") or node.get_content(metadata_mode="none")
            score = safe_float(getattr(source_node, "score", None))
            page_label = (
                metadata.get("page_label")
                or metadata.get("page_number")
                or metadata.get("page")
                or metadata.get("source")
            )
            citations.append(
                SourceCitation(
                    citation_id=f"S{idx}",
                    file_name=metadata.get("file_name"),
                    page_label=str(page_label) if page_label is not None else None,
                    score=score,
                    text=truncate_text(str(text), max_chars=1_400),
                )
            )
        except Exception:
            continue
    return citations


def action_trace_from_tool_event(event: ToolCallResult) -> str:
    tool_name = getattr(event, "tool_name", "tool")
    kwargs = getattr(event, "tool_kwargs", {}) or {}
    if tool_name == "financial_report_pdf":
        query = kwargs.get("input") or kwargs.get("query") or kwargs
        return f"Searched the uploaded PDF for: {truncate_text(str(query), 220)}"
    if tool_name == "market_data_search":
        ticker = kwargs.get("ticker") or kwargs
        return f"Fetched market data/news from yfinance for: {ticker}"
    return f"Used tool `{tool_name}` with input: {truncate_text(str(kwargs), 220)}"


def sanitize_react_stream_for_actions(stream_text: str) -> List[str]:
    """Extract safe action-level trace. Avoid displaying raw chain-of-thought text."""
    actions: List[str] = []
    for line in stream_text.splitlines():
        clean = line.strip()
        if clean.startswith("Action:"):
            tool = clean.split(":", 1)[1].strip()
            actions.append(f"Planned tool call: `{tool}`")
        elif clean.startswith("Action Input:"):
            tool_input = clean.split(":", 1)[1].strip()
            actions.append(f"Prepared tool input: {truncate_text(tool_input, 240)}")
    return actions


async def run_agent(agent: ReActAgent, question: str) -> AgentRunResult:
    ctx = Context(agent)
    handler = agent.run(question, ctx=ctx)

    raw_stream = ""
    action_trace: List[str] = []
    sources: List[SourceCitation] = []

    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            raw_stream += getattr(event, "delta", "") or ""
        elif isinstance(event, ToolCallResult):
            action_trace.append(action_trace_from_tool_event(event))
            tool_output = getattr(event, "tool_output", None)
            raw_output = getattr(tool_output, "raw_output", None)
            if raw_output is not None and hasattr(raw_output, "source_nodes"):
                sources.extend(extract_source_citations(raw_output.source_nodes))

    response = await handler

    # Some LlamaIndex response objects also attach source nodes at the final layer.
    if hasattr(response, "source_nodes"):
        sources.extend(extract_source_citations(response.source_nodes))

    # Fallback action parsing if ToolCallResult events were not exposed by the runtime.
    if not action_trace and raw_stream:
        action_trace = sanitize_react_stream_for_actions(raw_stream)

    # De-duplicate citations by text.
    seen = set()
    unique_sources: List[SourceCitation] = []
    for source in sources:
        key = (source.file_name, source.page_label, source.text[:200])
        if key not in seen:
            seen.add(key)
            source.citation_id = f"S{len(unique_sources) + 1}"
            unique_sources.append(source)

    return AgentRunResult(
        answer=str(response),
        action_trace=action_trace,
        sources=unique_sources[:6],
    )


# -----------------------------------------------------------------------------
# UI rendering
# -----------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 1.2rem; max-width: 1450px; }
        .small-muted { color: #6b7280; font-size: 0.88rem; }
        .metric-card {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            background: rgba(250, 250, 250, 0.45);
        }
        .source-card {
            border-left: 4px solid rgba(49, 51, 63, 0.25);
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.6rem;
            background: rgba(250, 250, 250, 0.45);
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> Tuple[Optional[SidebarConfig], Optional[Any]]:
    with st.sidebar:
        st.header("Configuration")
        env_key = os.getenv("OPENAI_API_KEY", "")
        openai_api_key = st.text_input(
            "OpenAI API key",
            type="password",
            value=st.session_state.get("openai_api_key", env_key),
            help="Stored only in Streamlit session state. You can also set OPENAI_API_KEY in your environment.",
        )
        st.session_state["openai_api_key"] = openai_api_key

        uploaded_pdf = st.file_uploader("Upload annual report / 10-K PDF", type=["pdf"])

        ticker = st.text_input("Dashboard ticker", value=st.session_state.get("ticker", "AAPL"))
        st.session_state["ticker"] = ticker

        with st.expander("Advanced settings", expanded=False):
            parser_strategy = st.selectbox(
                "Document parser",
                options=["sentence_window", "semantic_splitter"],
                index=0,
                help="Sentence window preserves nearby sentences; semantic splitter chunks by semantic breaks.",
            )
            similarity_top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=12, value=5)
            llm_model = st.text_input("LLM model", value="gpt-4o-mini")
            embedding_model = st.text_input("Embedding model", value="text-embedding-3-small")
            chart_period = st.selectbox("Chart period", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            chart_interval = st.selectbox("Chart interval", ["1d", "1wk", "1mo"], index=0)
            reset_index = st.checkbox("Force re-index PDF", value=False)

        try:
            config = SidebarConfig(
                openai_api_key=openai_api_key,
                ticker=ticker,
                llm_model=llm_model,
                embedding_model=embedding_model,
                parser_strategy=parser_strategy,
                similarity_top_k=similarity_top_k,
                chart_period=chart_period,
                chart_interval=chart_interval,
                reset_index=reset_index,
            )
        except ValidationError as exc:
            st.error(exc.errors()[0]["msg"])
            return None, uploaded_pdf

        if not openai_api_key:
            st.warning("Enter an OpenAI API key to build the RAG agent.")

        st.divider()
        st.caption("Data from yfinance may be delayed/rate-limited. This app is for research, not financial advice.")
        return config, uploaded_pdf


def render_market_panel(config: SidebarConfig) -> None:
    st.subheader(f"Market Dashboard: {config.ticker}")
    try:
        snapshot = StockSnapshot(**fetch_stock_snapshot_dict(config.ticker))
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Last price", "N/A" if snapshot.price is None else f"{snapshot.price:,.2f} {snapshot.currency or ''}")
        col_b.metric("Previous close", "N/A" if snapshot.previous_close is None else f"{snapshot.previous_close:,.2f}")
        col_c.metric("P/E", "N/A" if snapshot.pe_ratio is None else f"{snapshot.pe_ratio:,.2f}")

        hist = fetch_price_history(config.ticker, config.chart_period, config.chart_interval)
        if hist.empty:
            st.info("No price history returned for this ticker/period.")
        else:
            st.line_chart(hist, use_container_width=True)

        with st.expander("Recent news", expanded=False):
            if not snapshot.news:
                st.write("No recent yfinance news returned.")
            for item in snapshot.news[:5]:
                title = item.title
                if item.link:
                    st.markdown(f"- [{title}]({item.link})")
                else:
                    st.markdown(f"- {title}")
                meta = " | ".join(x for x in [item.publisher, item.published] if x)
                if meta:
                    st.caption(meta)

        if snapshot.warnings:
            with st.expander("Market data warnings", expanded=False):
                for warning in snapshot.warnings:
                    st.warning(warning)
    except Exception as exc:
        st.error(f"Could not load market dashboard for {config.ticker}: {exc}")


def render_messages() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("Thought Trace", expanded=False):
                    for step_idx, step in enumerate(msg["trace"], start=1):
                        st.write(f"{step_idx}. {step}")
            if msg.get("sources"):
                with st.expander("Source Citations", expanded=False):
                    for source in msg["sources"]:
                        render_source_card(SourceCitation(**source))


def render_source_card(source: SourceCitation) -> None:
    score_text = f" | similarity: {source.score:.3f}" if source.score is not None else ""
    page_text = f" | page: {html.escape(source.page_label)}" if source.page_label else ""
    file_text = html.escape(source.file_name or st.session_state.get("indexed_pdf_name") or "uploaded PDF")
    citation_id = html.escape(source.citation_id)
    source_text = html.escape(source.text)
    st.markdown(
        f"""
        <div class="source-card">
            <strong>[{citation_id}] {file_text}</strong>{page_text}{score_text}<br/>
            <span>{source_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_index_status(uploaded_pdf: Any, config: SidebarConfig) -> Tuple[Optional[Any], Optional[OpenAI]]:
    if uploaded_pdf is None:
        st.info("Upload a financial PDF in the sidebar to activate document RAG.")
        return None, None

    if not config.openai_api_key.get_secret_value():
        st.info("Add an OpenAI API key in the sidebar to index the PDF.")
        return None, None

    pdf_bytes = uploaded_pdf.getvalue()
    pdf_hash = file_sha256(pdf_bytes)
    pdf_path = save_uploaded_pdf(uploaded_pdf, pdf_hash)

    try:
        with st.status("Preparing financial report index...", expanded=False) as status:
            query_engine, node_count, collection_name = build_query_engine(pdf_path, pdf_hash, config)
            llm, _embed_model = configure_llamaindex(
                config.openai_api_key.get_secret_value(),
                config.llm_model,
                config.embedding_model,
            )
            st.session_state.indexed_pdf_hash = pdf_hash
            st.session_state.indexed_pdf_name = uploaded_pdf.name
            status.update(
                label=f"Indexed `{uploaded_pdf.name}` into Chroma collection `{collection_name}` with {node_count:,} nodes.",
                state="complete",
                expanded=False,
            )
        return query_engine, llm
    except Exception as exc:
        st.error(str(exc))
        return None, None


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    inject_css()
    init_session_state()

    config, uploaded_pdf = render_sidebar()
    if config is None:
        return

    st.title(APP_TITLE)
    st.markdown(f"<div class='small-muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)

    chat_col, market_col = st.columns([1.15, 0.85], gap="large")

    with market_col:
        render_market_panel(config)

    with chat_col:
        query_engine, llm = render_index_status(uploaded_pdf, config)
        st.subheader("Analyst Chat")
        render_messages()

        disabled = query_engine is None or llm is None
        prompt = st.chat_input(
            "Ask about the uploaded report, a ticker, or both...",
            disabled=disabled,
        )

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            agent = create_agent(query_engine, llm)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document and market data..."):
                    try:
                        result: AgentRunResult = run_coroutine_sync(run_agent(agent, prompt))
                    except Exception as exc:
                        result = AgentRunResult(
                            answer=f"I could not complete the analysis because the agent failed: {exc}",
                            action_trace=["The agent run failed before completion."],
                            sources=[],
                        )

                st.markdown(result.answer)

                with st.expander("Thought Trace", expanded=False):
                    if result.action_trace:
                        for step_idx, step in enumerate(result.action_trace, start=1):
                            st.write(f"{step_idx}. {step}")
                    else:
                        st.write("No external tool calls were recorded for this response.")

                if result.sources:
                    with st.expander("Source Citations", expanded=True):
                        for source in result.sources:
                            render_source_card(source)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "trace": result.action_trace,
                    "sources": [source.model_dump() for source in result.sources],
                }
            )


if __name__ == "__main__":
    main()
