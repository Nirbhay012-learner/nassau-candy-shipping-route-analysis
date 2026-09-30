
import io
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    from openpyxl.styles import Font, PatternFill, Alignment
    OPENPYXL_OK = True
except Exception:
    OPENPYXL_OK = False



# FILE / PROJECT CONFIG

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_shipping_data.csv"
ROUTE_PATH = PROJECT_ROOT / "data" / "route_performance.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "random_forest_cost_model.pkl"

sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.feature_engineering import FACTORY_MAPPING
except Exception:
    FACTORY_MAPPING = {}

st.set_page_config(
    page_title="Nassau Candy | Supply Chain Intelligence",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)



# UI / THEME

st.markdown(
    """
<style>
:root {
  --bg:#050a12; --panel:#0b1422; --panel2:#0f1b2d; --line:#20324a;
  --text:#edf5ff; --muted:#91a4bc; --accent:#64d8ff; --accent2:#8b7cff;
  --good:#5ee6a8; --warn:#ffd166; --bad:#ff7b8a;
}
.stApp {
  background:
    radial-gradient(circle at 90% 0%, rgba(100,216,255,.10), transparent 24%),
    radial-gradient(circle at 10% 20%, rgba(139,124,255,.08), transparent 25%),
    linear-gradient(135deg,#03070d 0%,#07111f 55%,#040911 100%);
  color:var(--text);
}
.main .block-container {
  max-width:1550px; padding-top:1.1rem; padding-bottom:3rem;
  animation: pageIn .45s ease-out;
}
@keyframes pageIn { from {opacity:0; transform:translateY(8px)} to {opacity:1; transform:none} }
@keyframes pulse { 0%,100%{opacity:.55} 50%{opacity:1} }
.hero {
  border:1px solid var(--line); border-radius:24px; padding:26px 30px;
  background:linear-gradient(135deg,rgba(16,30,49,.96),rgba(7,14,25,.92));
  box-shadow:0 20px 60px rgba(0,0,0,.22); margin-bottom:18px;
}
.hero h1 {margin:0; font-size:2.2rem; letter-spacing:-.04em;}
.hero p {color:var(--muted); margin:.45rem 0 0; font-size:1rem;}
.badge {
  display:inline-block; padding:5px 10px; border-radius:999px;
  border:1px solid rgba(100,216,255,.28); color:var(--accent);
  background:rgba(100,216,255,.07); font-size:.75rem; margin-bottom:10px;
}
.card {
  background:linear-gradient(145deg,rgba(15,27,45,.96),rgba(8,16,28,.96));
  border:1px solid var(--line); border-radius:18px; padding:18px;
  transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  box-shadow:0 8px 30px rgba(0,0,0,.16);
}
.card:hover {transform:translateY(-2px); border-color:#315170; box-shadow:0 12px 36px rgba(0,0,0,.25);}
.insight {
  border-left:3px solid var(--accent); background:rgba(100,216,255,.045);
  padding:13px 15px; border-radius:10px; margin-bottom:10px;
}
.insight b {color:#fff;}
.small {color:var(--muted); font-size:.84rem;}
.metric-card {min-height:115px;}
.metric-label {color:var(--muted); font-size:.82rem;}
.metric-value {font-size:1.55rem; font-weight:700; margin-top:6px;}
.metric-delta {font-size:.78rem; margin-top:5px;}
.good {color:var(--good)} .warn {color:var(--warn)} .bad {color:var(--bad)}
.section-title {font-size:1.25rem; font-weight:700; margin:22px 0 10px;}
.status-dot {
  display:inline-block; width:8px; height:8px; border-radius:50%;
  background:var(--good); box-shadow:0 0 10px rgba(94,230,168,.7);
  animation:pulse 1.8s infinite; margin-right:7px;
}
.footer {color:#657993; font-size:.78rem; text-align:center; padding:25px 0 5px;}
div[data-testid="stMetric"] {
  background:linear-gradient(145deg,rgba(15,27,45,.96),rgba(8,16,28,.96));
  border:1px solid var(--line); padding:14px 16px; border-radius:16px;
}
button[kind="primary"] {border-radius:12px;}

/* Executive polish */
.page-kicker{display:inline-flex;gap:7px;align-items:center;color:#7dd3fc;font-size:.72rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px}
.hero{position:relative;overflow:hidden}
.hero:after{content:"";position:absolute;width:260px;height:260px;right:-100px;top:-150px;border-radius:50%;background:rgba(100,216,255,.10);filter:blur(22px);animation:heroGlow 5s ease-in-out infinite}
@keyframes heroGlow{0%,100%{transform:scale(.9);opacity:.5}50%{transform:scale(1.18);opacity:1}}
.card{backdrop-filter:blur(10px)}
.metric-card{min-height:128px}
.kpi-accent{height:3px;border-radius:4px;background:linear-gradient(90deg,#64d8ff,#8b7cff);margin-top:10px;opacity:.65}
.insight{transition:transform .2s ease,background .2s ease}
.insight:hover{transform:translateX(3px);background:rgba(100,216,255,.07)}
.watch{border:1px solid rgba(255,209,102,.18);background:rgba(255,209,102,.045);border-radius:14px;padding:12px 14px;margin-bottom:8px}
.ai-panel{border:1px solid rgba(139,124,255,.22);background:linear-gradient(145deg,rgba(25,22,52,.78),rgba(8,15,29,.95));border-radius:20px;padding:20px;box-shadow:0 18px 50px rgba(0,0,0,.22)}
.ask-chip{display:inline-block;border:1px solid rgba(100,216,255,.18);background:rgba(100,216,255,.05);border-radius:999px;padding:5px 9px;color:#9ddff5;font-size:.72rem;margin:3px}
</style>
""",
    unsafe_allow_html=True,
)



# DATA LOADING

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_PATH)
    for c in ["Order Date", "Ship Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_routes():
    if ROUTE_PATH.exists():
        return pd.read_csv(ROUTE_PATH)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


try:
    with st.spinner("Loading supply-chain intelligence..."):
        df = load_data()
        route_df = load_routes()
        model = load_model()
except Exception as exc:
    st.error("The dashboard could not load its project files.")
    st.code(str(exc))
    st.info(
        "Check that data/processed_shipping_data.csv, "
        "data/random_forest_cost_model.pkl and src/feature_engineering.py exist."
    )
    st.stop()



# HELPERS

def money(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "—"


def safe_mean(frame, col):
    return float(frame[col].mean()) if col in frame.columns and len(frame) else 0.0


def pct_change(current, previous):
    if previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def fmt_delta(v):
    if v is None:
        return "—"
    return f"{v:+.1f}%"


def card_metric(label, value, delta=None, help_text=None):
    delta_html = ""
    if delta is not None:
        cls = "good" if delta >= 0 else "bad"
        delta_html = f'<div class="metric-delta {cls}">{fmt_delta(delta)} vs comparison period</div>'
    st.markdown(
        f"""
        <div class="card metric-card" title="{help_text or ''}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          {delta_html}<div class="kpi-accent"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(message, hint="Try changing the active filters."):
    st.markdown(
        f"""
        <div class="card" style="text-align:center;padding:34px">
          <div style="font-size:2rem">◌</div>
          <b>{message}</b>
          <div class="small" style="margin-top:6px">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_plot(fig, height=390):
    if not PLOTLY_OK:
        st.warning("Plotly is not installed. Run: pip install plotly")
        return
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbeafe"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(145,164,188,.12)")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def export_excel(frames):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, frame in frames.items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])
        if OPENPYXL_OK:
            wb = writer.book
            for ws in wb.worksheets:
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill("solid", fgColor="12233A")
                    cell.alignment = Alignment(horizontal="center")
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    max_len = min(max(len(str(c.value or "")) for c in col) + 2, 34)
                    ws.column_dimensions[col[0].column_letter].width = max_len
    return output.getvalue()


def export_pdf(summary, table_frame):
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Nassau Candy — Supply Chain Intelligence Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(summary, styles["BodyText"]),
        Spacer(1, 12),
    ]
    rows = [list(table_frame.columns)] + table_frame.astype(str).values.tolist()
    rows = rows[:21]
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#12233A")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#B7C4D6")),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(table)
    doc.build(story)
    return buf.getvalue()


def destination_state_code(value):
    mapping = {
        "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
        "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
        "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
        "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
        "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS",
        "Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH",
        "New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC",
        "North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA",
        "Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN",
        "Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA",
        "West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","District of Columbia":"DC"
    }
    return mapping.get(str(value), None)


def filtered_data(base):
    out = base.copy()
    factory = st.session_state.get("gf", "All")
    destination = st.session_state.get("gs", "All")
    mode = st.session_state.get("gm", "All")
    if factory != "All" and "Factory" in out.columns:
        out = out[out["Factory"] == factory]
    if destination != "All" and "State/Province" in out.columns:
        out = out[out["State/Province"] == destination]
    if mode != "All" and "Ship Mode" in out.columns:
        out = out[out["Ship Mode"] == mode]
    if "Order Date" in out.columns and st.session_state.get("date_range"):
        date_value = st.session_state.date_range
        if isinstance(date_value, (tuple, list)) and len(date_value) == 2:
            start, end = date_value
            out = out[out["Order Date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return out



# SIDEBAR / GLOBAL FILTERS

with st.sidebar:
    st.markdown("## 🍫 Nassau Candy")
    st.caption("Supply Chain Intelligence")
    st.markdown('<span class="status-dot"></span> Dashboard online', unsafe_allow_html=True)
    st.divider()

    pages = [
        "🏠 Executive Overview",
        "💡 Executive Insights",
        "🛣️ Route Intelligence",
        "🚚 Shipping Analysis",
        "🚨 Anomaly Detection",
        "🤖 AI Cost Predictor",
        "🧪 ML Performance",
        "🔎 Ask the Data",
        "📦 Data Explorer",
        "📤 Reports & Exports",
        "ℹ️ About",
    ]
    page = st.radio("Navigate", pages)

    st.divider()
    st.markdown("### Global filters")

    factories = ["All"] + sorted(df["Factory"].dropna().astype(str).unique().tolist())
    destinations = ["All"] + sorted(df["State/Province"].dropna().astype(str).unique().tolist())
    modes = ["All"] + sorted(df["Ship Mode"].dropna().astype(str).unique().tolist())

    if "gf" not in st.session_state:
        st.session_state.gf = "All"
    if "gs" not in st.session_state:
        st.session_state.gs = "All"
    if "gm" not in st.session_state:
        st.session_state.gm = "All"

    st.selectbox("Factory", factories, key="gf")
    st.selectbox("Destination", destinations, key="gs")
    st.selectbox("Shipping Mode", modes, key="gm")

    if "Order Date" in df.columns and df["Order Date"].notna().any():
        dmin = df["Order Date"].min().date()
        dmax = df["Order Date"].max().date()
        if "date_range" not in st.session_state:
            st.session_state.date_range = (dmin, dmax)
        st.date_input(
            "Order date range",
            min_value=dmin,
            max_value=dmax,
            key="date_range",
        )

    def reset_filters():
        st.session_state.gf = "All"
        st.session_state.gs = "All"
        st.session_state.gm = "All"
        if "Order Date" in df.columns and df["Order Date"].notna().any():
            st.session_state.date_range = (dmin, dmax)

    st.button("↺ Reset filters", width="stretch", on_click=reset_filters)

    st.divider()
    st.caption(f"Rows available: {len(df):,}")
    st.caption(f"Last loaded: {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}")


filt = filtered_data(df)

# Previous-period comparison based on equal-length date window.
comparison = pd.DataFrame()
if "Order Date" in filt.columns and filt["Order Date"].notna().any() and len(filt):
    dates = filt["Order Date"].dropna()
    if len(dates):
        start, end = dates.min(), dates.max()
        span = max((end - start).days + 1, 1)
        prev_start, prev_end = start - pd.Timedelta(days=span), start - pd.Timedelta(days=1)
        comparison = df.copy()
        # Preserve all global filters except the current date window for a valid prior-period comparison.
        if st.session_state.get("gf", "All") != "All":
            comparison = comparison[comparison["Factory"] == st.session_state.gf]
        if st.session_state.get("gs", "All") != "All":
            comparison = comparison[comparison["State/Province"] == st.session_state.gs]
        if st.session_state.get("gm", "All") != "All":
            comparison = comparison[comparison["Ship Mode"] == st.session_state.gm]
        comparison = comparison[comparison["Order Date"].between(prev_start, prev_end)]



# HERO

st.markdown(
    """
<div class="hero">
  <div class="page-kicker">🍫 NASSAU CANDY • SUPPLY CHAIN INTELLIGENCE</div>
  <div class="badge">AI + ANALYTICS • EXECUTIVE COMMAND CENTER</div>
  <h1>Factory → Customer Shipping Intelligence</h1>
  <p>Monitor volume, cost, route concentration, shipping behavior and model-driven shipment cost estimates from one decision-ready workspace.</p>
</div>
""",
    unsafe_allow_html=True,
)



# EXECUTIVE OVERVIEW

if page == "🏠 Executive Overview":
    st.markdown("### Executive command center")
    st.caption("A decision-ready summary of the currently filtered dataset.")

    if filt.empty:
        empty_state("No records match the current filters.")
        st.stop()

    current = {
        "Shipments": len(filt),
        "Sales": filt["Sales"].sum(),
        "Cost": filt["Cost"].sum(),
        "Gross Profit": filt["Gross Profit"].sum(),
        "Avg Lead Time": safe_mean(filt, "Lead Time"),
        "Avg Cost": safe_mean(filt, "Cost"),
    }
    previous = {}
    if len(comparison):
        previous = {
            "Shipments": len(comparison),
            "Sales": comparison["Sales"].sum(),
            "Cost": comparison["Cost"].sum(),
            "Gross Profit": comparison["Gross Profit"].sum(),
            "Avg Lead Time": safe_mean(comparison, "Lead Time"),
            "Avg Cost": safe_mean(comparison, "Cost"),
        }

    cols = st.columns(6)
    vals = [
        ("Shipments", f"{current['Shipments']:,}"),
        ("Sales", money(current["Sales"])),
        ("Shipping Cost", money(current["Cost"])),
        ("Gross Profit", money(current["Gross Profit"])),
        ("Avg Lead Time", f"{current['Avg Lead Time']:.2f} days"),
        ("Avg Ship Cost", money(current["Avg Cost"])),
    ]
    for col, (label, value) in zip(cols, vals):
        with col:
            d = pct_change(current[label], previous.get(label)) if previous else None
            card_metric(label, value, d)

    st.markdown("### Business highlights")
    highlights = []
    if "Ship Mode" in filt.columns and len(filt):
        mode = filt["Ship Mode"].value_counts().idxmax()
        share = filt["Ship Mode"].value_counts(normalize=True).max() * 100
        highlights.append(("Shipping pattern", f"{mode} is the most frequent mode, representing {share:.1f}% of filtered shipments."))
    if "Factory" in filt.columns and len(filt):
        factory = filt.groupby("Factory")["Order ID"].count().idxmax()
        highlights.append(("Factory volume", f"{factory} handles the highest filtered shipment count."))
    if "State/Province" in filt.columns and len(filt):
        dest = filt.groupby("State/Province")["Cost"].mean().idxmax()
        cost = filt.groupby("State/Province")["Cost"].mean().max()
        highlights.append(("Cost concentration", f"{dest} has the highest average shipping cost in the current selection ({money(cost)})."))
    for title, text_ in highlights:
        st.markdown(f'<div class="insight"><b>{title}</b><br>{text_}</div>', unsafe_allow_html=True)

    st.markdown("### Executive insights")
    i1, i2, i3 = st.columns(3)
    top_dest = filt["State/Province"].value_counts().index[0] if len(filt) else "—"
    top_dest_share = filt["State/Province"].value_counts(normalize=True).iloc[0]*100 if len(filt) else 0
    top_factory = filt["Factory"].value_counts().index[0] if len(filt) else "—"
    top_factory_share = filt["Factory"].value_counts(normalize=True).iloc[0]*100 if len(filt) else 0
    with i1: st.markdown(f'<div class="card"><div class="metric-label">Destination concentration</div><div class="metric-value">{top_dest_share:.1f}%</div><div class="small">{top_dest} is the largest destination by shipment volume.</div></div>', unsafe_allow_html=True)
    with i2: st.markdown(f'<div class="card"><div class="metric-label">Factory concentration</div><div class="metric-value">{top_factory_share:.1f}%</div><div class="small">{top_factory} is the largest factory by shipment volume.</div></div>', unsafe_allow_html=True)
    with i3: st.markdown(f'<div class="card"><div class="metric-label">Cost / sales ratio</div><div class="metric-value">{(current["Cost"]/current["Sales"]*100 if current["Sales"] else 0):.1f}%</div><div class="small">Shipping cost as a share of filtered sales.</div></div>', unsafe_allow_html=True)

    st.markdown("### Operational watchlist")
    route_watch = filt.groupby(["Factory","State/Province"]).agg(Shipments=("Order ID","count"), Avg_Cost=("Cost","mean"), Avg_Lead_Time=("Lead Time","mean")).reset_index()
    route_watch = route_watch[route_watch["Shipments"] >= max(3, int(route_watch["Shipments"].quantile(.75)))]
    route_watch = route_watch.sort_values(["Avg_Cost","Avg_Lead_Time"], ascending=False).head(5)
    if len(route_watch):
        st.dataframe(route_watch, width="stretch", hide_index=True)

    st.markdown("### Performance trends")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        if PLOTLY_OK and "Order Date" in filt.columns:
            trend = filt.dropna(subset=["Order Date"]).groupby(pd.Grouper(key="Order Date", freq="ME")).agg(
                Sales=("Sales","sum"), Cost=("Cost","sum")
            ).reset_index()
            fig = px.line(trend, x="Order Date", y=["Sales","Cost"], markers=True, title="Sales and shipping cost over time")
            make_plot(fig)
        else:
            st.line_chart(filt.set_index("Order Date")[["Sales","Cost"]])
    with c2:
        mode_counts = filt["Ship Mode"].value_counts().reset_index()
        mode_counts.columns = ["Ship Mode", "Shipments"]
        if PLOTLY_OK:
            fig = px.pie(mode_counts, names="Ship Mode", values="Shipments", hole=.58, title="Shipping mode mix")
            make_plot(fig, 390)
        else:
            st.bar_chart(mode_counts.set_index("Ship Mode"))

    c1, c2 = st.columns(2)
    with c1:
        fac = filt.groupby("Factory").agg(Shipments=("Order ID","count"), Sales=("Sales","sum"), Cost=("Cost","sum")).reset_index()
        if PLOTLY_OK:
            fig = px.bar(fac.sort_values("Shipments", ascending=False).head(10), x="Factory", y="Shipments", title="Top factories by shipment volume")
            make_plot(fig)
        else:
            st.bar_chart(fac.set_index("Factory")["Shipments"])
    with c2:
        dest = filt.groupby("State/Province").agg(Shipments=("Order ID","count"), Avg_Cost=("Cost","mean")).reset_index().sort_values("Shipments", ascending=False).head(10)
        if PLOTLY_OK:
            fig = px.bar(dest, x="Shipments", y="State/Province", orientation="h", title="Top destinations")
            make_plot(fig)
        else:
            st.dataframe(dest, width="stretch", hide_index=True)



# EXECUTIVE INSIGHTS

elif page == "💡 Executive Insights":
    st.markdown("### Executive insights")
    st.caption("Automatically generated observations from the currently filtered data. These are descriptive, not causal claims.")

    if filt.empty:
        empty_state("No data available for insights.")
    else:
        # Cost / volume concentration
        route = filt.groupby(["Factory","State/Province"]).agg(
            Shipments=("Order ID","count"), Cost=("Cost","sum"), Avg_Cost=("Cost","mean"),
            Lead_Time=("Lead Time","mean")
        ).reset_index()
        top_route = route.sort_values("Shipments", ascending=False).iloc[0]
        expensive_route = route.sort_values("Avg_Cost", ascending=False).iloc[0]
        cost_corr = filt[["Sales","Units","Cost"]].corr(numeric_only=True)["Cost"].drop("Cost").sort_values(key=np.abs, ascending=False)
        mode = filt["Ship Mode"].value_counts().idxmax()
        mode_share = filt["Ship Mode"].value_counts(normalize=True).max() * 100

        insights = [
            ("📦 Volume concentration", f"The busiest factory-to-destination combination is <b>{top_route['Factory']} → {top_route['State/Province']}</b> with {int(top_route['Shipments']):,} shipments."),
            ("💰 Cost concentration", f"The highest average shipping cost route in the current selection is <b>{expensive_route['Factory']} → {expensive_route['State/Province']}</b> at {money(expensive_route['Avg_Cost'])}."),
            ("🚚 Shipping mix", f"<b>{mode}</b> is the dominant shipping mode at {mode_share:.1f}% of filtered shipments."),
            ("📊 Cost relationship", f"The strongest linear numeric association with Cost in this dataset slice is <b>{cost_corr.index[0]}</b> (correlation {cost_corr.iloc[0]:.3f})."),
            ("🧠 ML interpretation", "The trained Random Forest places most of its reported importance on Sales, while Ship Mode has very low importance in the current model."),
        ]
        for title, text_ in insights:
            st.markdown(f'<div class="insight"><b>{title}</b><br>{text_}</div>', unsafe_allow_html=True)

        st.markdown("### Operational watchlist")
        watch = route[route["Shipments"] >= max(2, int(route["Shipments"].quantile(.75)))]
        watch = watch.sort_values(["Avg_Cost","Lead_Time"], ascending=False).head(10)
        if watch.empty:
            empty_state("No watchlist routes meet the current volume threshold.")
        else:
            st.dataframe(watch, width="stretch", hide_index=True)



# ROUTE INTELLIGENCE

elif page == "🛣️ Route Intelligence":
    st.markdown("### Route intelligence")
    st.caption("Explore route volume, cost, lead time and factory-to-destination relationships.")

    if filt.empty:
        empty_state("No routes match the current filters.")
    else:
        route = filt.groupby(["Factory","State/Province"]).agg(
            Shipments=("Order ID","count"),
            Sales=("Sales","sum"),
            Units=("Units","sum"),
            Cost=("Cost","sum"),
            Avg_Cost=("Cost","mean"),
            Avg_Lead_Time=("Lead Time","mean"),
            Gross_Profit=("Gross Profit","sum"),
        ).reset_index()

        a,b,c,d = st.columns(4)
        a.metric("Active routes", f"{len(route):,}")
        b.metric("Shipments", f"{len(filt):,}")
        c.metric("Avg route cost", money(route["Avg_Cost"].mean()))
        d.metric("Avg lead time", f"{route['Avg_Lead_Time'].mean():.2f} days")

        t1, t2, t3 = st.tabs(["Route Overview", "Network", "Route Drilldown"])
        with t1:
            if PLOTLY_OK:
                top = route.sort_values("Shipments", ascending=False).head(15).copy()
                top["Route"] = top["Factory"] + " → " + top["State/Province"]
                fig = px.bar(top, x="Shipments", y="Route", orientation="h", color="Avg_Cost", title="Highest-volume routes")
                make_plot(fig, 500)
            st.dataframe(route.sort_values("Shipments", ascending=False), width="stretch", hide_index=True)
            st.download_button("Download route CSV", route.to_csv(index=False).encode(), "route_analysis.csv", "text/csv")

        with t2:
            st.markdown("#### Factory → destination network")
            network = route.sort_values("Shipments", ascending=False).head(30).copy()
            if PLOTLY_OK:
                network["Route"] = network["Factory"] + " → " + network["State/Province"]
                fig = go.Figure()
                for factory, group in network.groupby("Factory"):
                    fig.add_trace(go.Bar(name=factory, x=group["State/Province"], y=group["Shipments"]))
                fig.update_layout(barmode="stack", title="Network shipment concentration")
                make_plot(fig, 480)
            st.dataframe(network[["Factory","State/Province","Shipments","Avg_Cost","Avg_Lead_Time"]], width="stretch", hide_index=True)

            # Geographic view
            geo = filt.groupby("State/Province").agg(Shipments=("Order ID","count"), Avg_Cost=("Cost","mean")).reset_index()
            geo["Code"] = geo["State/Province"].map(destination_state_code)
            geo = geo.dropna(subset=["Code"])
            st.markdown("#### Geographic destination view")
            if PLOTLY_OK and len(geo):
                fig = px.choropleth(
                    geo, locations="Code", locationmode="USA-states", color="Shipments",
                    hover_name="State/Province", hover_data={"Avg_Cost":":.2f","Code":False},
                    scope="usa", title="Shipment volume by destination state"
                )
                make_plot(fig, 520)
            else:
                empty_state("A geographic map needs recognizable U.S. state names in State/Province.")

        with t3:
            fac = st.selectbox("Drill into factory", sorted(filt["Factory"].dropna().astype(str).unique()))
            dests = sorted(filt.loc[filt["Factory"].astype(str) == fac, "State/Province"].dropna().astype(str).unique())
            if not dests:
                empty_state("No destinations for this factory.")
            else:
                dest = st.selectbox("Drill into destination", dests)
                z = filt[(filt["Factory"].astype(str) == fac) & (filt["State/Province"].astype(str) == dest)]
                a,b,c,d = st.columns(4)
                a.metric("Shipments", f"{len(z):,}")
                b.metric("Sales", money(z["Sales"].sum()))
                c.metric("Shipping cost", money(z["Cost"].sum()))
                d.metric("Avg lead time", f"{safe_mean(z,'Lead Time'):.2f} days")
                if PLOTLY_OK and len(z):
                    mode_z = z.groupby("Ship Mode").size().reset_index(name="Shipments")
                    fig = px.bar(mode_z, x="Ship Mode", y="Shipments", title=f"Shipping modes — {fac} → {dest}")
                    make_plot(fig)
                st.dataframe(z.head(100), width="stretch", hide_index=True)



# SHIPPING ANALYSIS

elif page == "🚚 Shipping Analysis":
    st.markdown("### Shipping analysis")
    st.caption("Compare shipping modes across volume, cost intensity and dataset-derived lead time.")
    if filt.empty:
        empty_state("No shipments match the current filters.")
    else:
        ss = filt.groupby("Ship Mode").agg(
            Shipments=("Order ID","count"), Sales=("Sales","sum"), Units=("Units","sum"),
            Cost=("Cost","sum"), Avg_Cost=("Cost","mean"), Avg_Lead_Time=("Lead Time","mean"),
            Gross_Profit=("Gross Profit","sum")
        ).reset_index()
        ss["Volume Share %"] = ss["Shipments"] / ss["Shipments"].sum() * 100
        ss["Cost Share %"] = ss["Cost"] / ss["Cost"].sum() * 100
        ss["Cost / Sales %"] = ss["Cost"] / ss["Sales"] * 100

        a,b,c,d = st.columns(4)
        a.metric("Most used mode", ss.loc[ss["Shipments"].idxmax(),"Ship Mode"])
        b.metric("Highest avg cost", ss.loc[ss["Avg_Cost"].idxmax(),"Ship Mode"])
        c.metric("Lowest avg cost", ss.loc[ss["Avg_Cost"].idxmin(),"Ship Mode"])
        d.metric("Modes compared", f"{len(ss)}")

        t1,t2,t3 = st.tabs(["Mode comparison","Cost vs volume","Detailed table"])
        with t1:
            if PLOTLY_OK:
                fig = px.bar(ss.sort_values("Shipments"), x="Shipments", y="Ship Mode", orientation="h", title="Shipment volume by shipping mode", text="Volume Share %")
                make_plot(fig, 390)
                fig = px.bar(ss, x="Ship Mode", y="Avg_Lead_Time", title="Average lead time by mode")
                make_plot(fig, 390)
        with t2:
            if PLOTLY_OK:
                fig = px.scatter(ss, x="Shipments", y="Avg_Cost", size="Cost", color="Ship Mode", hover_data=["Avg_Lead_Time","Cost / Sales %"], title="Cost vs shipment volume")
                make_plot(fig, 430)
            st.info("Lead Time is calculated directly from the source dates. The source data produces unusually large values, so interpret this metric as dataset-derived.")
        with t3:
            st.dataframe(ss.sort_values("Shipments", ascending=False), width="stretch", hide_index=True)
            st.download_button("Download shipping analysis", ss.to_csv(index=False).encode(), "shipping_analysis.csv", "text/csv")



# AI COST PREDICTOR

elif page == "🚨 Anomaly Detection":
    st.markdown("### Anomaly detection")
    st.caption("Rule-based screening for unusually expensive shipments and unusual route behavior. No model is retrained.")
    if filt.empty:
        empty_state("No shipments available for anomaly screening.")
    else:
        c1,c2 = st.columns(2)
        q1 = filt["Cost"].quantile(.75)
        iqr = q1 - filt["Cost"].quantile(.25)
        cost_cutoff = q1 + 1.5*iqr
        high_cost = filt[filt["Cost"] > cost_cutoff].copy().sort_values("Cost", ascending=False)
        route_an = filt.groupby(["Factory","State/Province"]).agg(Shipments=("Order ID","count"), Avg_Cost=("Cost","mean"), Avg_Lead_Time=("Lead Time","mean"), Total_Cost=("Cost","sum")).reset_index()
        min_volume = max(3, int(route_an["Shipments"].quantile(.50)))
        route_pool = route_an[route_an["Shipments"] >= min_volume].copy()
        if len(route_pool):
            cost_q = route_pool["Avg_Cost"].quantile(.75)
            lead_q = route_pool["Avg_Lead_Time"].quantile(.75)
            unusual_routes = route_pool[(route_pool["Avg_Cost"] >= cost_q) | (route_pool["Avg_Lead_Time"] >= lead_q)].copy().sort_values(["Avg_Cost","Avg_Lead_Time"], ascending=False)
        else:
            unusual_routes = route_pool
        with c1:
            st.markdown(f'<div class="card"><div class="metric-label">High-cost threshold</div><div class="metric-value">{money(cost_cutoff)}</div><div class="small">IQR rule: Q3 + 1.5 × IQR</div></div>', unsafe_allow_html=True)
            st.metric("Unusually expensive shipments", f"{len(high_cost):,}")
        with c2:
            st.markdown(f'<div class="card"><div class="metric-label">Route screening volume</div><div class="metric-value">{min_volume:,}</div><div class="small">Minimum shipments used to reduce tiny-route noise.</div></div>', unsafe_allow_html=True)
            st.metric("Unusual routes", f"{len(unusual_routes):,}")

        a,b=st.tabs(["Expensive shipments","Unusual routes"])
        with a:
            if len(high_cost):
                selected = st.selectbox("Inspect shipment", high_cost.index.tolist(), format_func=lambda x: f"Row {x} • {money(high_cost.loc[x,'Cost'])}")
                r = high_cost.loc[selected]
                x1,x2,x3,x4 = st.columns(4)
                x1.metric("Cost", money(r["Cost"]))
                x2.metric("Sales", money(r["Sales"]))
                x3.metric("Units", f"{r['Units']:g}")
                x4.metric("Ship Mode", str(r["Ship Mode"]))
                st.dataframe(high_cost.head(100), width="stretch", hide_index=True)
            else:
                empty_state("No unusually expensive shipments detected.", "The current filtered data contains no records above the IQR threshold.")
        with b:
            if len(unusual_routes):
                st.dataframe(unusual_routes.head(50), width="stretch", hide_index=True)
            else:
                empty_state("No unusual routes detected for the current selection.")

elif page == "🤖 AI Cost Predictor":
    st.markdown("### AI shipping-cost predictor")
    st.caption("Enter a hypothetical shipment. The prediction uses the trained Random Forest model saved in the project.")

    c1,c2 = st.columns(2)
    products = sorted(df["Product Name"].dropna().astype(str).unique())
    factories_model = sorted(df["Factory"].dropna().astype(str).unique())
    states_model = sorted(df["State/Province"].dropna().astype(str).unique())
    modes_model = sorted(df["Ship Mode"].dropna().astype(str).unique())
    divisions = sorted(df["Division"].dropna().astype(str).unique())

    with c1:
        product = st.selectbox("Product", products)
        mapped_factory = FACTORY_MAPPING.get(product, factories_model[0] if factories_model else "Unknown")
        st.text_input("Factory (from product mapping)", value=mapped_factory, disabled=True)
        factory = mapped_factory
        state = st.selectbox("Destination", states_model)
        mode = st.selectbox("Shipping Mode", modes_model)
    with c2:
        division = st.selectbox("Division", divisions)
        units = st.number_input("Units", min_value=1.0, value=3.0, step=1.0)
        sales = st.number_input("Sales", min_value=0.0, value=10.0, step=0.5)
        predict = st.button("✨ Predict shipping cost", type="primary", width="stretch")

    region_mode = df.loc[df["State/Province"] == state, "Region"].dropna().mode()
    region = region_mode.iloc[0] if len(region_mode) else "Unknown"

    if predict:
        row = pd.DataFrame([{
            "Product Name": product, "Factory": factory, "State/Province": state,
            "Region": region, "Ship Mode": mode, "Division": division,
            "Units": units, "Sales": sales
        }])
        try:
            pred = float(model.predict(row)[0])
            st.markdown(
                f"""
                <div class="ai-panel" style="text-align:center">
                  <div class="badge">MODEL OUTPUT</div>
                  <div class="small">Estimated shipping cost</div>
                  <div style="font-size:3.4rem;font-weight:800;letter-spacing:-.05em">{money(pred)}</div>
                  <div class="small">For the exact inputs shown below</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            s1,s2,s3 = st.columns(3)
            s1.metric("Sales input", money(sales))
            s2.metric("Units input", f"{units:g}")
            s3.metric("Shipping mode", mode)

            with st.expander("🔍 Why did the model produce this prediction?", expanded=True):
                st.write(
                    "The reported feature importance from the trained model indicates that Sales "
                    "is the dominant numeric driver, followed by Units and selected categorical features. "
                    "Ship Mode has very low reported importance, so changing it alone may not materially "
                    "change a prediction."
                )
                imp = pd.DataFrame({
                    "Feature": ["Sales","Units","Division","Product","Factory","Ship Mode"],
                    "Importance": [0.899284,0.049829,0.032310,0.008836,0.003430,0.000112]
                })
                if PLOTLY_OK:
                    fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Reported model feature importance")
                    make_plot(fig, 330)
        except Exception as exc:
            st.error("Prediction failed for these inputs.")
            st.code(str(exc))
            st.info("The input columns must match the preprocessing used when the model was trained.")



# ML PERFORMANCE

elif page == "🧪 ML Performance":
    st.markdown("### Machine-learning performance")
    a,b,c,d = st.columns(4)
    a.metric("Evaluation MAE", "0.00636")
    b.metric("Evaluation RMSE", "0.08487")
    c.metric("Evaluation R²", "0.99962")
    d.metric("Mean CV R²", "0.99249")

    st.caption("These values are the reported project evaluation results. They should be interpreted with the dataset and validation design used during training.")

    t1,t2,t3 = st.tabs(["Accuracy", "Feature Importance", "Sensitivity"])
    with t1:
        if "Cost" in filt.columns and PLOTLY_OK:
            # A fresh prediction comparison is only attempted when the model can consume the raw frame.
            sample = filt.head(1000).copy()
            try:
                X = sample.drop(columns=["Cost"], errors="ignore")
                y = sample["Cost"]
                pred = model.predict(X)
                compare = pd.DataFrame({"Actual": y.values, "Predicted": pred})
                fig = px.scatter(compare, x="Actual", y="Predicted", title="Actual vs predicted shipping cost")
                mn = float(min(compare.min()))
                mx = float(max(compare.max()))
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx)
                make_plot(fig, 500)
            except Exception:
                st.info("The saved model's preprocessing pipeline is not directly compatible with the displayed raw dataframe. The reported evaluation metrics remain available above.")
    with t2:
        imp = pd.DataFrame({
            "Feature": ["Sales","Units","Division_Other","Product_Kazookles","Factory_The_Other_Factory","Ship_Mode"],
            "Importance": [0.899284,0.049829,0.032310,0.008836,0.003430,0.000112]
        }).sort_values("Importance")
        if PLOTLY_OK:
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Random Forest feature importance")
            make_plot(fig, 430)
        st.dataframe(imp.sort_values("Importance", ascending=False), width="stretch", hide_index=True)
    with t3:
        st.markdown(
            '<div class="insight"><b>Shipping-mode sensitivity</b><br>'
            'The controlled project test produced the same predicted cost (2.28) for First Class, '
            'Same Day, Second Class and Standard Class while holding the other inputs fixed. '
            'This is consistent with the model having extremely low reported importance for Ship Mode.'
            '</div>',
            unsafe_allow_html=True,
        )


# ANOMALY DETECTION — also surfaced through Ask the Data

elif page == "🔎 Ask the Data":
    st.markdown("### Ask the Data")
    st.caption("Choose a question. Answers are computed directly from the filtered dataframe.")

    questions = [
        "Which factory has the highest shipping cost?",
        "Which destination has the highest average shipping cost?",
        "Which shipping mode is most common?",
        "Which route has the highest shipment volume?",
        "Which products have the highest average shipping cost?",
        "Show unusual high-cost shipments",
        "Which factory has the longest average lead time?",
        "How concentrated are shipments across destinations?",
    ]
    st.markdown("<span class='ask-chip'>volume</span><span class='ask-chip'>cost</span><span class='ask-chip'>routes</span><span class='ask-chip'>shipping modes</span><span class='ask-chip'>anomalies</span>", unsafe_allow_html=True)
    q = st.selectbox("Question", questions)

    if filt.empty:
        empty_state("No data is available for the selected question.")
    elif q == questions[0]:
        g = filt.groupby("Factory")["Cost"].mean().sort_values(ascending=False)
        st.success(f"{g.index[0]} has the highest average shipping cost at {money(g.iloc[0])}.")
        st.dataframe(g.rename("Average Cost").reset_index(), width="stretch", hide_index=True)
    elif q == questions[1]:
        g = filt.groupby("State/Province")["Cost"].mean().sort_values(ascending=False)
        st.success(f"{g.index[0]} has the highest average shipping cost at {money(g.iloc[0])}.")
        st.dataframe(g.rename("Average Cost").reset_index(), width="stretch", hide_index=True)
    elif q == questions[2]:
        g = filt["Ship Mode"].value_counts()
        st.success(f"{g.index[0]} is the most common shipping mode with {int(g.iloc[0]):,} shipments.")
        st.dataframe(g.rename("Shipments").reset_index(), width="stretch", hide_index=True)
    elif q == questions[3]:
        g = filt.groupby(["Factory","State/Province"]).size().sort_values(ascending=False)
        st.success(f"{g.index[0][0]} → {g.index[0][1]} has the highest shipment volume: {int(g.iloc[0]):,}.")
        st.dataframe(g.rename("Shipments").reset_index(), width="stretch", hide_index=True)
    elif q == questions[4]:
        g = filt.groupby("Product Name")["Cost"].mean().sort_values(ascending=False).head(15)
        st.dataframe(g.rename("Average Cost").reset_index(), width="stretch", hide_index=True)
    elif q == questions[5]:
        q1 = filt["Cost"].quantile(.75)
        iqr = filt["Cost"].quantile(.75) - filt["Cost"].quantile(.25)
        cutoff = q1 + 1.5 * iqr
        anomalies = filt[filt["Cost"] > cutoff].copy().sort_values("Cost", ascending=False)
        st.success(f"{len(anomalies):,} records are above the IQR-based high-cost threshold of {money(cutoff)}.")
        st.dataframe(anomalies.head(50), width="stretch", hide_index=True)
    elif q == questions[6]:
        g = filt.groupby("Factory")["Lead Time"].mean().sort_values(ascending=False)
        st.success(f"{g.index[0]} has the longest average lead time at {g.iloc[0]:.2f} days.")
        st.dataframe(g.rename("Average Lead Time").reset_index(), width="stretch", hide_index=True)
    else:
        counts = filt["State/Province"].value_counts()
        top_share = counts.iloc[0] / counts.sum() * 100
        st.info(f"The largest destination accounts for {top_share:.1f}% of filtered shipments.")
        st.dataframe(counts.rename("Shipments").reset_index(), width="stretch", hide_index=True)



# DATA EXPLORER

elif page == "📦 Data Explorer":
    st.markdown("### Data explorer")
    divs = ["All"] + sorted(df["Division"].dropna().astype(str).unique())
    div = st.selectbox("Division", divs)
    view = filt if div == "All" else filt[filt["Division"].astype(str) == div]
    a,b,c = st.columns(3)
    a.metric("Rows", f"{len(view):,}")
    b.metric("Columns", f"{view.shape[1]:,}")
    c.metric("Missing values", f"{int(view.isna().sum().sum()):,}")
    st.dataframe(view, width="stretch", height=560, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode(), "filtered_shipping_data.csv", "text/csv")



# REPORTS / EXPORTS

elif page == "📤 Reports & Exports":
    st.markdown("### Reports & exports")
    st.caption("Create a snapshot of the current filtered dataset for presentation or further analysis.")

    summary = (
        f"Filtered shipments: {len(filt):,}. "
        f"Sales: {money(filt['Sales'].sum())}. "
        f"Shipping cost: {money(filt['Cost'].sum())}. "
        f"Average lead time: {safe_mean(filt,'Lead Time'):.2f} days."
    )
    st.markdown(f'<div class="insight"><b>Current report scope</b><br>{summary}</div>', unsafe_allow_html=True)

    route_export = filt.groupby(["Factory","State/Province"]).agg(
        Shipments=("Order ID","count"), Sales=("Sales","sum"), Cost=("Cost","sum"),
        Avg_Cost=("Cost","mean"), Avg_Lead_Time=("Lead Time","mean")
    ).reset_index()
    mode_export = filt.groupby("Ship Mode").agg(
        Shipments=("Order ID","count"), Sales=("Sales","sum"), Cost=("Cost","sum"),
        Avg_Lead_Time=("Lead Time","mean")
    ).reset_index()

    frames = {"Filtered_Data": filt, "Routes": route_export, "Shipping_Modes": mode_export}
    excel_bytes = export_excel(frames)
    st.download_button(
        "📊 Download Excel report",
        excel_bytes,
        "nassau_candy_supply_chain_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

    pdf_bytes = export_pdf(summary, route_export)
    if pdf_bytes:
        st.download_button("📄 Download PDF report", pdf_bytes, "nassau_candy_supply_chain_report.pdf", "application/pdf")
    else:
        st.warning("PDF export needs reportlab. Run: pip install reportlab")

    st.markdown("### Included sheets")
    st.write("• Filtered_Data  • Routes  • Shipping_Modes")



# ABOUT

elif page == "ℹ️ About":
    st.markdown("### About this project")
    st.markdown(
        """
        <div class="card">
        <b>Nassau Candy Supply Chain Intelligence</b><br><br>
        An end-to-end analytics dashboard combining exploratory analysis,
        route intelligence, operational summaries and machine-learning shipping-cost prediction.
        <br><br>
        <b>Pipeline</b><br>
        Raw shipping data → preprocessing / feature engineering → ML model → interactive dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Reported project model results")
    a,b,c = st.columns(3)
    a.metric("Random Forest MAE", "0.00636")
    b.metric("Random Forest RMSE", "0.08487")
    c.metric("Random Forest R²", "0.99962")
    st.caption("Reported evaluation values from the project analysis.")

st.markdown('<div class="footer">Nassau Candy Supply Chain Intelligence • Built with Streamlit, Pandas, Plotly and scikit-learn</div>', unsafe_allow_html=True)
