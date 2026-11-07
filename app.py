"""Corvit Systems Rawalpindi - Student Assistant (Home + AI chatbot + Courses & Fees)."""
import csv
import datetime
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).parent
MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
UNKNOWN_TAG = "[[UNKNOWN]]"
LOG_FILE = BASE / "unanswered.csv"
RED, DARK = "#C8102E", "#8E0B21"
LEVELS = ["Basic", "Fundamental", "Associate", "Professional", "Expert"]

CAMPUS_NAME = "Rawalpindi Campus"
ADDRESS = "2nd Floor Zarwar Center, Opp Rabi Center, Murree Road, Rawalpindi"
PHONE = "051 4928004"
TIMINGS = "9 AM to 9 PM"
MAP_URL = "https://maps.google.com/maps?q=corvit%20systems%20rawalpindi&t=&z=13&ie=UTF8&iwloc=&output=embed"

st.set_page_config(page_title="Corvit Systems Rawalpindi", page_icon="🎓", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp, button, input, textarea {{ font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important; }}
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}
.stApp {{ background: #FFFFFF; color: #1F2937; }}
.block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 1220px; }}

/* top utility bar */
.topbar {{ background: #1F2937; color: #E5E7EB; padding: 9px 22px; font-size: .8rem; font-weight: 500;
          display: flex; justify-content: space-between; border-radius: 0 0 10px 10px; letter-spacing: .2px; }}
.topbar b {{ color: #fff; }}

/* brand + navigation */
.brand {{ display: flex; align-items: center; gap: 12px; padding-top: 10px; }}
.brand .mark {{ width: 42px; height: 42px; background: {RED}; color: #fff; border-radius: 10px; display: flex;
               align-items: center; justify-content: center; font-weight: 800; font-size: 1.3rem; }}
.brand .name {{ font-weight: 800; font-size: 1.3rem; color: #111827; line-height: 1.1; }}
.brand .name small {{ display: block; font-size: .68rem; color: {RED}; font-weight: 700; letter-spacing: 2px; margin-top: 2px; }}
div[role="radiogroup"] {{ gap: 4px; justify-content: flex-end; padding-top: 12px; }}
div[role="radiogroup"] label {{ padding: 8px 16px 6px 16px; border-bottom: 3px solid transparent; border-radius: 0; cursor: pointer; }}
div[role="radiogroup"] label > div:first-child {{ display: none; }}
div[role="radiogroup"] label p {{ color: #374151; font-weight: 600; font-size: .95rem; margin: 0; }}
div[role="radiogroup"] label:hover p {{ color: {RED}; }}
div[role="radiogroup"] label:has(input:checked) {{ border-bottom-color: {RED}; }}
div[role="radiogroup"] label:has(input:checked) p {{ color: {RED}; }}
.navline {{ height: 1px; background: #E5E7EB; margin: -6px 0 22px 0; }}

/* hero */
.hero {{ background: linear-gradient(115deg, {RED} 0%, {DARK} 100%); color: #fff; padding: 30px 40px;
        border-radius: 16px; margin-bottom: 22px; position: relative; overflow: hidden; }}
.hero:after {{ content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px;
              border-radius: 50%; background: rgba(255,255,255,.07); }}
.hero .badge {{ display: inline-block; background: rgba(255,255,255,.16); padding: 5px 14px; border-radius: 999px;
               font-size: .75rem; font-weight: 700; letter-spacing: 1.5px; }}
.hero h1 {{ color: #fff !important; font-size: 2.2rem; font-weight: 800; margin: 14px 0 8px 0; padding: 0; line-height: 1.15; }}
.hero p {{ font-size: 1.05rem; opacity: .95; margin: 0; max-width: 640px; line-height: 1.55; }}

/* stat cards */
.stat {{ border: 1px solid #E5E7EB; border-top: 3px solid {RED}; border-radius: 12px; padding: 16px 18px;
        background: #fff; box-shadow: 0 1px 3px rgba(17,24,39,.06); }}
.stat b {{ display: block; font-size: 1.55rem; color: #111827; font-weight: 800; }}
.stat span {{ color: #6B7280; font-size: .82rem; font-weight: 500; }}

/* section titles */
.section-title {{ font-size: 1.25rem; font-weight: 700; color: #111827; margin: 6px 0 14px 0; }}
.section-title:after {{ content: ""; display: block; width: 42px; height: 3px; background: {RED}; margin-top: 7px; border-radius: 2px; }}
.page-title {{ font-size: 1.9rem; font-weight: 800; color: #111827; margin: 6px 0 2px 0; }}
.page-sub {{ color: #6B7280; margin-bottom: 18px; }}

/* chat */
.chat-head {{ background: #FFF5F6; border: 1px solid #E5E7EB; border-bottom: none; border-radius: 12px 12px 0 0;
             padding: 12px 18px; font-weight: 700; color: #111827; display: flex; align-items: center; gap: 10px; }}
.chat-head .dot {{ width: 9px; height: 9px; border-radius: 50%; background: #16A34A; }}
.chat-head small {{ margin-left: auto; color: #6B7280; font-weight: 500; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 0 0 12px 12px; border-color: #E5E7EB; }}
[data-testid="stChatMessage"] {{ background: #F9FAFB; border: 1px solid #EEF0F3; border-radius: 12px; }}
[data-testid="stChatInput"] {{ border-color: {RED}; }}

/* buttons (FAQ list) */
.stButton > button {{ background: #fff; color: #1F2937; border: 1px solid #E5E7EB; border-radius: 10px;
                     text-align: left; font-weight: 500; padding: 10px 14px; justify-content: flex-start; }}
.stButton > button:hover {{ border-color: {RED}; color: {RED}; background: #FFF5F6; }}

/* ask form */
[data-testid="stForm"] {{ border: 1px solid #E5E7EB; border-top: none; border-bottom: none; border-radius: 0; padding: 12px 14px 0 14px; background: #fff; }}
[data-testid="stTextInput"] input {{ border: 1.5px solid #E5E7EB; border-radius: 10px; padding: 11px 14px; }}
[data-testid="stTextInput"] input:focus {{ border-color: {RED}; box-shadow: 0 0 0 1px {RED}; }}
[data-testid="stFormSubmitButton"] button {{ background: {RED}; color: #fff; border: 1px solid {RED}; border-radius: 10px;
                                           font-weight: 700; justify-content: center; padding: 10px 14px; }}
[data-testid="stFormSubmitButton"] button:hover {{ background: {DARK}; border-color: {DARK}; color: #fff; }}

/* info cards */
.card {{ border: 1px solid #E5E7EB; border-left: 4px solid {RED}; border-radius: 12px; padding: 16px 18px;
        background: #fff; box-shadow: 0 1px 3px rgba(17,24,39,.05); margin-top: 14px; }}
.card h4 {{ color: #111827; margin: 0 0 8px 0; font-size: 1rem; }}
.card p {{ margin: 0; color: #4B5563; font-size: .92rem; line-height: 1.6; }}

/* footer */
.footer {{ background: #1F2937; color: #D1D5DB; padding: 20px 26px; border-radius: 14px; margin-top: 34px;
          display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-size: .85rem; }}
.footer b {{ color: #fff; }}
</style>
""",
    unsafe_allow_html=True,
)


# ================================================================ data
@st.cache_data
def load_courses() -> pd.DataFrame:
    return pd.read_csv(BASE / "courses.csv")


@st.cache_data
def load_info() -> list[str]:
    text = (BASE / "info.md").read_text(encoding="utf-8")
    chunks = []
    for block in text.split("\n## "):
        lines = [l.strip() for l in block.splitlines() if l.strip() and "TODO" not in l]
        if len(lines) >= 2:
            chunks.append("\n".join(lines).lstrip("# "))
    return chunks


def course_text(r) -> str:
    return (
        f"Course: {r.course} | Category: {r.field} | Vendor: {r.vendor} | Level: {r.level} | "
        f"In-campus regular fee: PKR {r.campus_regular_pkr:,} | In-campus one-to-one fee: PKR {r.campus_one_to_one_pkr:,} | "
        f"Online group fee: ${r.online_group_usd:,} | Online one-to-one fee: ${r.online_one_to_one_usd:,}"
    )


@st.cache_data
def catalog_text() -> str:
    return "\n".join(course_text(r) for r in load_courses().itertuples())


def md_table(df: pd.DataFrame) -> str:
    rows = ["| Course | Level | Campus Regular (PKR) | Campus One-to-One (PKR) | Online Group ($) | Online One-to-One ($) |",
            "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        rows.append(f"| {r.course} | {r.level} | {r.campus_regular_pkr:,} | {r.campus_one_to_one_pkr:,} | "
                    f"{r.online_group_usd:,} | {r.online_one_to_one_usd:,} |")
    return "\n".join(rows)


# ================================================================ backend search
@st.cache_resource
def row_index():
    df = load_courses()
    txt = (df.course + " " + df.field + " " + df.vendor + " " + df.level).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return vec, vec.fit_transform(txt)


@st.cache_resource
def info_index():
    docs = load_info()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    return docs, vec, vec.fit_transform(docs)


def _num(s: str, k) -> float:
    v = float(s.replace(",", ""))
    return v * 1000 if k else v


def parse_budget(q: str):
    """Detect price limits like 'under 40,000', 'below 30k', 'above $500'."""
    q = q.lower()
    usd = bool(re.search(r"\$|usd|dollar|online", q))
    num = r"(?:rs\.?|pkr|\$)?\s*([\d,]+(?:\.\d+)?)\s*(k)?"
    lo = hi = None
    m = re.search(r"(?:under|below|less than|within|upto|up to|max|maximum|cheaper than|budget of|budget)\s*" + num, q)
    if m:
        hi = _num(m.group(1), m.group(2))
    m = re.search(r"(?:above|over|more than|at least|minimum|min)\s*" + num, q)
    if m:
        lo = _num(m.group(1), m.group(2))
    return lo, hi, usd


def backend_search(query: str, k: int = 12) -> pd.DataFrame:
    df = load_courses().copy()
    vec, mat = row_index()
    df["score"] = cosine_similarity(vec.transform([query]), mat)[0]
    lo, hi, usd = parse_budget(query)
    col = "online_group_usd" if usd else "campus_regular_pkr"
    if lo is not None:
        df = df[df[col] >= lo]
    if hi is not None:
        df = df[df[col] <= hi]
    if lo is None and hi is None:
        df = df[df.score > 0.15]
    if re.search(r"cheap|lowest|affordable|low cost|least", query.lower()):
        df = df.sort_values([col, "score"], ascending=[True, False])
    else:
        df = df.sort_values(["score", col], ascending=[False, True])
    return df.head(k)


def retrieve_info(query: str, k: int = 4) -> list[str]:
    docs, vec, mat = info_index()
    sims = cosine_similarity(vec.transform([query]), mat)[0]
    return [docs[i] for i in sims.argsort()[::-1][:k] if sims[i] > 0.05]


# ================================================================ AI
def get_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.getenv("ANTHROPIC_API_KEY")


def ask_claude(system: str, messages: list[dict], max_tokens: int = 900) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=get_key())
    resp = client.messages.create(model=MODEL, max_tokens=max_tokens, system=system, messages=messages)
    return "".join(b.text for b in resp.content if b.type == "text")


def build_system(results: pd.DataFrame, info: list[str]) -> str:
    found = "\n".join(course_text(r) for r in results.itertuples()) or "(no direct match)"
    ctx = "\n".join(f"- {c}" for c in info) or "(none)"
    return f"""You are "Corvit Assistant", the student support assistant of Corvit Systems, Rawalpindi Campus.
Rules:
- Answer ONLY from BACKEND SEARCH RESULTS, CATALOG and CONTEXT below. Never invent fees, dates, batch timings, addresses or phone numbers.
- You only have information about the Rawalpindi campus. If asked about another campus or city, say so politely and suggest corvit.com.pk.
- Always reply in simple, professional, friendly English (even if the user writes Urdu or Roman Urdu). Keep answers short and clear. Use a markdown table or bullets when listing courses with fees.
- In-campus fees are in PKR, online fees are in USD (2025 Rawalpindi fee structure). When giving prices, add a short note that fees may change and should be confirmed with the campus.
- If the answer is not available below, say you do not have that information, suggest contacting the Rawalpindi campus on {PHONE}, and end your reply with {UNKNOWN_TAG}.
- For recommendations, suggest 2-3 matching courses with a one-line reason each.
- Politely decline questions not related to Corvit, its courses, fees or the Rawalpindi campus.

BACKEND SEARCH RESULTS (courses matched to the student's question):
{found}

CONTEXT (institute information):
{ctx}

CATALOG (all courses):
{catalog_text()}"""


def log_unknown(question: str):
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "question"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), question])


# ================================================================ FAQ answers (instant, no AI needed)
def faq_courses() -> str:
    df = load_courses()
    lines = [f"- **{f}** - {n} courses" for f, n in df.groupby("field", sort=False).size().items()]
    return (f"Corvit Rawalpindi offers **{len(df)} courses** in these categories:\n\n" + "\n".join(lines)
            + "\n\nOpen the **Courses & Fees** page to see every course with its fee.")


def faq_ai() -> str:
    df = load_courses()
    d = df[(df.field == "Artificial Intelligence") | (df.course == "Python")]
    return "Here are the AI and Python courses:\n\n" + md_table(d) + "\n\n*Fees may change - please confirm with the campus.*"


def faq_fee_types() -> str:
    return ("Every course is available in 4 modes:\n\n"
            "- **In-Campus Regular Class** (group) - fee in PKR\n"
            "- **In-Campus One-to-One Priority Class** - fee in PKR\n"
            "- **Online Group Class** - fee in USD\n"
            "- **Online One-to-One Class** - fee in USD\n\n"
            "Ask me about any course (for example *\"CCNA fee\"*) and I will show all four fees.")


def faq_cheapest() -> str:
    d = load_courses().sort_values(["campus_regular_pkr", "course"]).head(8)
    return "The most affordable courses (in-campus regular class):\n\n" + md_table(d) + "\n\n*Fees may change - please confirm with the campus.*"


def faq_timings() -> str:
    return (f"Corvit Systems working timings are **{TIMINGS}**.\n\n"
            "The Rawalpindi batch schedule is published at corvit.com.pk/rawalpindi/. "
            f"Please confirm the current batch timing with the campus on {PHONE}.")


def faq_location() -> str:
    return f"**{CAMPUS_NAME}**\n\n- Address: {ADDRESS}\n- Phone: {PHONE}\n- Timings: {TIMINGS}"


def faq_certificate() -> str:
    return "Yes. According to the Corvit website, students **earn a recognized certificate upon course completion**."


def faq_contact() -> str:
    return (f"You can reach the **{CAMPUS_NAME}** here:\n\n- Phone: {PHONE}\n- Address: {ADDRESS}\n- Timings: {TIMINGS}\n"
            "- Website: corvit.com.pk\n- Facebook: CorvitSystems | Instagram: corvit_systems")


FAQS = {
    "What courses does Corvit offer?": faq_courses,
    "Which AI courses are available?": faq_ai,
    "What are the fee options?": faq_fee_types,
    "What are the cheapest courses?": faq_cheapest,
    "What are the working timings?": faq_timings,
    "Where is the Rawalpindi campus?": faq_location,
    "Do students get a certificate?": faq_certificate,
    "How can I contact Corvit?": faq_contact,
}


def answer(q: str, history: list[dict]) -> str:
    if q in FAQS:
        return FAQS[q]()
    results = backend_search(q)
    info = retrieve_info(q)
    if not get_key():
        body = ("Here is what I found:\n\n" + md_table(results)) if not results.empty else \
               ("\n".join(f"- {c}" for c in info) or f"I could not find that. Please contact the campus on {PHONE}.")
        return body + "\n\n*(AI key not set - showing search results only.)*"
    hist = history[-6:] + [{"role": "user", "content": q}]
    while hist and hist[0]["role"] != "user":
        hist.pop(0)
    try:
        text = ask_claude(build_system(results, info), hist)
    except Exception as e:
        return f"Sorry, I could not reach the AI service ({type(e).__name__}). Please try again."
    if UNKNOWN_TAG in text:
        text = text.replace(UNKNOWN_TAG, "").strip()
        log_unknown(q)
    return text


# ================================================================ layout
def header():
    st.markdown(
        f'<div class="topbar"><span>Working timings: <b>{TIMINGS}</b></span>'
        f'<span>{CAMPUS_NAME} &nbsp;|&nbsp; Phone: <b>{PHONE}</b> &nbsp;|&nbsp; corvit.com.pk</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([5, 4])
    c1.markdown('<div class="brand"><div class="mark">C</div><div class="name">CORVIT SYSTEMS'
                '<small>RAWALPINDI CAMPUS</small></div></div>', unsafe_allow_html=True)
    with c2:
        st.radio("Navigation", ["Home", "Courses & Fees", "Contact"], horizontal=True,
                 key="nav", label_visibility="collapsed")
    st.markdown('<div class="navline"></div>', unsafe_allow_html=True)


def footer():
    st.markdown(
        f'<div class="footer"><span><b>Corvit Systems - {CAMPUS_NAME}</b><br>{ADDRESS}</span>'
        f'<span>Phone: {PHONE}<br>Working timings: {TIMINGS}</span>'
        '<span>Fees shown are the 2025 Rawalpindi fee structure.<br>Please confirm with the campus before admission.</span></div>',
        unsafe_allow_html=True,
    )


def set_pending(q: str):
    st.session_state.pending = q


def stat_row(items):
    for col, (big, small) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="stat"><b>{big}</b><span>{small}</span></div>', unsafe_allow_html=True)


def page_home():
    df = load_courses()
    st.markdown(
        '<div class="hero"><span class="badge">RAWALPINDI CAMPUS</span><h1>Reaching End to End</h1>'
        '<p>Empowering students through modern instruction and supportive learning. '
        'Ask our AI assistant about courses, fees and admission at Corvit Systems Rawalpindi.</p></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown('<div class="section-title">Frequently asked questions</div>', unsafe_allow_html=True)
        for i, q in enumerate(FAQS):
            st.button(q, key=f"faq{i}", on_click=set_pending, args=(q,), width="stretch")

    with left:
        st.markdown('<div class="section-title">Ask Corvit Assistant</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-head"><span class="dot"></span>Corvit Assistant'
                    '<small>Courses, fees and admission</small></div>', unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content":
                "Hello! I'm the Corvit Assistant for the Rawalpindi campus. Ask me about courses, fees or admission, "
                "or choose a common question from the list."}]
        msgs = st.session_state.messages

        # question box sits at the TOP of the chat, so students never need to scroll to type
        with st.form("ask_form", clear_on_submit=True, border=False):
            c1, c2 = st.columns([6, 1])
            typed = c1.text_input("Your question", placeholder="Type your question here, e.g. Python course fee",
                                  label_visibility="collapsed")
            sent = c2.form_submit_button("Ask", width="stretch")
        q = (typed or "").strip() if sent else None
        q = q or st.session_state.pop("pending", None)

        box = st.container(height=430, border=True)
        with box:
            if q:  # newest exchange is shown first
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Searching Corvit data..."):
                        reply = answer(q, msgs)
                    st.markdown(reply)
            for pair in reversed([msgs[i:i + 2] for i in range(1, len(msgs), 2)]):
                for m in pair:
                    with st.chat_message(m["role"]):
                        st.markdown(m["content"])
            with st.chat_message(msgs[0]["role"]):
                st.markdown(msgs[0]["content"])
        if q:
            st.session_state.messages += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]

    st.write("")
    stat_row([("25+", "Years of excellence"), (str(len(df)), "Courses available"),
              (str(df.field.nunique()), "Course categories"), ("9 AM - 9 PM", "Working timings")])
    st.markdown(f'<div class="card"><h4>{CAMPUS_NAME}</h4><p>{ADDRESS}<br>Phone: {PHONE}</p></div>',
                unsafe_allow_html=True)


def page_courses():
    st.markdown('<div class="page-title">Courses & Fees</div>'
                '<div class="page-sub">2025 fee structure for the Rawalpindi campus. '
                'Official fee page: corvit.com.pk/rawalpindi-fees/</div>', unsafe_allow_html=True)
    df = load_courses()
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    cats = c1.multiselect("Category", sorted(df.field.unique()))
    lvls = c2.multiselect("Level", LEVELS)
    mode = c3.selectbox("Class type", ["All fees", "In-Campus (PKR)", "Online (USD)"])
    search = c4.text_input("Search course or vendor", placeholder="e.g. CCNA, Python, Huawei")
    if cats:
        df = df[df.field.isin(cats)]
    if lvls:
        df = df[df.level.isin(lvls)]
    if search:
        df = df[df.course.str.contains(search, case=False) | df.vendor.str.contains(search, case=False)]

    if df.empty:
        st.warning("No courses match your filters.")
        return
    stat_row([(str(len(df)), "Courses shown"), (str(df.field.nunique()), "Categories"),
              (f"PKR {df.campus_regular_pkr.min():,}", "In-campus fee from"),
              (f"${df.online_group_usd.min():,}", "Online group fee from")])
    st.write("")

    cols = {"field": "Category", "course": "Course", "vendor": "Vendor", "level": "Level"}
    fee_cols = {
        "campus_regular_pkr": "Campus Regular (PKR)", "campus_one_to_one_pkr": "Campus One-to-One (PKR)",
        "online_group_usd": "Online Group ($)", "online_one_to_one_usd": "Online One-to-One ($)",
    }
    if mode == "In-Campus (PKR)":
        fee_cols = {k: v for k, v in fee_cols.items() if "campus" in k}
    elif mode == "Online (USD)":
        fee_cols = {k: v for k, v in fee_cols.items() if "online" in k}
    show = df[list(cols) + list(fee_cols)].rename(columns={**cols, **fee_cols})
    st.dataframe(
        show, hide_index=True, width="stretch", height=540,
        column_config={v: st.column_config.NumberColumn(v, format="%d") for v in fee_cols.values()},
    )
    st.caption("Fees may change. Please confirm with the campus before admission.")


def page_contact():
    st.markdown(f'<div class="page-title">Contact</div><div class="page-sub">{CAMPUS_NAME}</div>',
                unsafe_allow_html=True)
    a, b = st.columns([2, 3], gap="large")
    with a:
        st.markdown(f'<div class="card"><h4>Address</h4><p>{ADDRESS}</p></div>'
                    f'<div class="card"><h4>Phone</h4><p>{PHONE}</p></div>'
                    f'<div class="card"><h4>Working timings</h4><p>{TIMINGS}</p></div>'
                    '<div class="card"><h4>Online</h4><p>Website: corvit.com.pk<br>Facebook: CorvitSystems<br>'
                    'Instagram: corvit_systems<br>YouTube: @corvitsystemsnorth</p></div>',
                    unsafe_allow_html=True)
    with b:
        try:
            if hasattr(st, "iframe"):
                st.iframe(MAP_URL, height=470)
            else:
                components.iframe(MAP_URL, height=470)
        except Exception:
            st.info("Map could not be loaded.")


header()
{"Home": page_home, "Courses & Fees": page_courses, "Contact": page_contact}[st.session_state.get("nav", "Home")]()
footer()
