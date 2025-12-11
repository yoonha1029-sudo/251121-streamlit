import io
import json
import textwrap
from typing import Dict, Any, List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import datetime

# ---- OpenAI SDK 확인 ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================
# API 키 (코드 내 삽입)
# =========================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")


# =========================
# [신규] 지식 파일 로드 헬퍼 (Simplified RAG)
# =========================
@st.cache_data # 앱 실행 시 한 번만 읽도록 캐시
def load_knowledge_file(file_path):
    """app.py와 동일한 위치에 있는 .txt 지식 파일을 읽습니다."""
    try:
        # GitHub 저장소의 루트에서 파일을 찾음
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"경고: 지식 파일({file_path})을 찾을 수 없습니다. AI가 일반적인 답변만 할 수 있습니다.")
        return ""
    except Exception as e:
        st.error(f"지식 파일 로드 오류: {e}")
        return ""

# --- 앱 시작 시 지식 파일 로드 ---
KNOWLEDGE_CURRICULUM = load_knowledge_file("knowledge_curriculum.txt")
KNOWLEDGE_DISASTERS = load_knowledge_file("knowledge_disasters.txt")


# =========================
# 페이지 기본 설정
# =========================
st.set_page_config(
    page_title="AI 기반 빅데이터 탐구 (홈)", 
    page_icon="🛰️",
    layout="wide",
)

# =========================
# 세션 상태 초기화
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []
if "df" not in st.session_state:
    st.session_state.df: Optional[pd.DataFrame] = None
if "api_key" not in st.session_state:
    st.session_state.api_key = OPENAI_API_KEY
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"
if "chart_spec" not in st.session_state:
    st.session_state.chart_spec: Optional[Dict[str, Any]] = None


# =========================
# 사이드바: AI 모델 설정
# =========================
with st.sidebar:
    st.markdown("## ⚙️ AI 모델 설정")
    if st.session_state.api_key == "YOUR_OPENAI_API_KEY_HERE" or not st.session_state.api_key:
        st.error("코드 상단의 OPENAI_API_KEY 변수에 실제 키를 입력하세요.")
    else:
        st.success("OpenAI API Key가 로드되었습니다.")
    st.session_state.model = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0,
        help="해석 정확도가 중요하면 상위 모델, 비용이 중요하면 mini 권장",
    )
    st.divider()
    st.info("데이터 다운로드는 'data' 페이지를 참고하세요.")


# =========================
# 상단 헤더
# =========================
st.title("🛰️ 재해·재난과 안전 빅데이터 탐구 지원 챗봇")
st.markdown(
    "중학생 과학 ‘재해·재난과 안전’ 수업에서 **빅데이터 탐구**를 돕는 챗봇입니다. "
    "데이터를 시각화하고, **AI에게 해석**을 요청해 보세요."
)
if st.session_state.api_key == "YOUR_OPENAI_API_KEY_HERE" or not st.session_state.api_key:
    st.error("분석을 시작하기 전에 Streamlit 코드의 `OPENAI_API_KEY` 변수에 실제 OpenAI API 키를 입력해야 합니다.")
    st.stop()


# =========================
# 1) 데이터 불러오기
# =========================
st.markdown("## 1) 데이터 불러오기 📥")
file = st.file_uploader(
    "CSV 또는 XLSX 파일 업로드",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    help="첫 번째 시트 기준(XLSX). 수업용 데이터는 'data' 페이지에서 다운로드 받으세요.",
)
def load_dataframe(_file) -> pd.DataFrame:
    if _file is None: return pd.DataFrame()
    if _file.name.lower().endswith(".csv"):
        try: df = pd.read_csv(_file, sep=",", low_memory=False, encoding='utf-8')
        except UnicodeDecodeError: df = pd.read_csv(_file, sep=",", low_memory=False, encoding='cp949')
    else: df = pd.read_excel(_file, engine="openpyxl")
    return df
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


# =========================
# 시각화 헬퍼 함수
# =========================
TIME_LIKE_KEYWORDS = ["연도", "년도", "year", "Year", "주", "week"]


def pick_time_like_column(df: pd.DataFrame) -> Optional[str]:
    # 1) dtype이 datetime 계열인 열
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # 2) 이름에 시간/연도/주차 관련 키워드가 포함된 열
    for col in df.columns:
        if any(k.lower() in col.lower() for k in TIME_LIKE_KEYWORDS):
            return col
    return None


def pick_numeric_column(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != exclude]
    return numeric_cols[0] if numeric_cols else None


def infer_chart(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    간단한 규칙 기반 차트 추천.
    returns (x, y, size, chart_type_label)
    """
    x_auto = pick_time_like_column(df)
    y_auto = pick_numeric_column(df, exclude=x_auto)

    # x 기준으로 차트 유형 판단
    chart_label = "선(line)"
    if x_auto is None and y_auto is not None:
        # 시간축 없으면 가장 단순한 막대/산점도로
        x_auto = df.columns[0]
        chart_label = "막대(bar)"
    elif x_auto is None and y_auto is None:
        chart_label = "막대(bar)"
    else:
        if x_auto and (pd.api.types.is_datetime64_any_dtype(df[x_auto]) or any(k.lower() in x_auto.lower() for k in TIME_LIKE_KEYWORDS)):
            chart_label = "선(line)"
        elif y_auto is not None:
            # 범주 x + 수치 y -> 막대
            chart_label = "막대(bar)"
        else:
            chart_label = "산점도(scatter)"

    return x_auto, y_auto, None, chart_label


def auto_describe_trend(df: pd.DataFrame, x: str, y: str) -> str:
    """
    간단한 규칙 기반 추세 설명 (2~3문장).
    """
    if x not in df.columns or y not in df.columns:
        return ""
    series = df[y].dropna()
    if series.empty or not pd.api.types.is_numeric_dtype(series):
        return ""
    first, last = series.iloc[0], series.iloc[-1]
    direction = last - first
    trend = "증가" if direction > 0 else "감소" if direction < 0 else "변화가 거의 없음"

    # 변동성 확인
    diff = series.diff().dropna()
    if not diff.empty:
        pos_ratio = (diff > 0).mean()
        neg_ratio = (diff < 0).mean()
    else:
        pos_ratio = neg_ratio = 0

    variability = ""
    if pos_ratio > 0.2:
        variability = "전체적으로 값이 증가하는 경향이 있습니다."
    elif neg_ratio > 0.2:
        variability = "전체적으로 값이 감소하는 경향이 있습니다."
    else:
        variability = "값의 변동 폭이 크고, 뚜렷한 증가/감소 경향은 보이지 않습니다."

    direction_text = f"처음 값({first:.2f}) 대비 마지막 값({last:.2f})이 {'높아졌습니다' if direction > 0 else '낮아졌습니다' if direction < 0 else '비슷합니다'}."
    return f"{variability} {direction_text}"
if file:
    df = load_dataframe(file)
    df = optimize_dtypes(df)
    st.session_state.df = df
if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    st.success(f"불러온 데이터: {df.shape[0]:,}행 × {df.shape[1]:,}열")
    with st.expander("📋 데이터 미리보기(상위 100행)", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)
    st.markdown("### 🔎 빠른 요약")
    col_meta1, col_meta2, col_meta3 = st.columns(3)
    with col_meta1: st.metric("행 수", f"{df.shape[0]:,}")
    with col_meta2: st.metric("열 수", f"{df.shape[1]:,}")
    with col_meta3:
        missing_total = int(df.isna().sum().sum())
        st.metric("결측치 총합", f"{missing_total:,}")
    with st.expander("🧮 기술통계(수치형)"):
        st.dataframe(df.describe().T, use_container_width=True)
    with st.expander("🧾 열 타입 정보"):
        info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(), "unique": df.nunique()})
        st.dataframe(info, use_container_width=True)
else:
    st.info("왼쪽 사이드바에서 **[data]** 페이지를 클릭해 CSV 파일을 다운로드 받거나, 가지고 있는 파일을 업로드하여 탐구를 시작하세요.")
    st.stop()


# =========================
# 2) 데이터 시각화
# =========================
st.markdown("## 2) 데이터 시각화 📊")
st.caption("핵심 차트 유형만 선택하고, AI와 함께 해석에 집중해 보세요.")
auto_mode = st.checkbox("🔀 자동 차트 추천 사용", value=True, help="데이터에서 시간/연도/주차/수치 열을 찾아 자동으로 차트를 만듭니다.")

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 자동 추천 실행
auto_x, auto_y, auto_size, auto_chart_label = infer_chart(df)
if auto_mode:
    st.info(
        f"추천 결과: 차트 유형='{auto_chart_label}', X축='{auto_x}', Y축='{auto_y if auto_y else '없음'}'"
    )
    if auto_y is None:
        st.warning("수치형 열을 찾지 못했습니다. 필요하면 자동 모드를 끄고 직접 Y축을 선택하세요.")

chart_type = st.selectbox(
    "차트 유형",
    ["선(line)", "막대(bar)", "산점도(scatter)", "원(pie)", "지도 (위도/경도)"],
    index=["선(line)", "막대(bar)", "산점도(scatter)", "원(pie)", "지도 (위도/경도)"].index(auto_chart_label) if auto_mode else 0,
    disabled=auto_mode
)

if chart_type.startswith("원("):
    x_label = "이름 (범주 열)"; y_label = "값 (수치 열)"; size_label = "추가 범례 (선택)"
elif chart_type.startswith("지도"):
    x_label = "위도 (Latitude) 열"; y_label = "경도 (Longitude) 열"; size_label = "크기/강도 (Magnitude) 열"
else: 
    x_label = "X축"; y_label = "Y축 (필요시)"; size_label = "크기 (선택, 산점도용)"

viz_col1, viz_col2, viz_col3 = st.columns(3)
with viz_col1:
    x_col = st.selectbox(
        x_label,
        options=all_cols,
        index=all_cols.index(auto_x) if auto_mode and auto_x in all_cols else 0,
        disabled=auto_mode and auto_x is not None
    )
with viz_col2:
    y_options = ["- 선택 안함 -"] + (numeric_cols if numeric_cols else all_cols)
    y_default = auto_y if auto_mode and auto_y in y_options else "- 선택 안함 -"
    y_col = st.selectbox(
        y_label,
        options=y_options,
        index=y_options.index(y_default) if y_default in y_options else 0,
        help="수치형 열을 우선 보여줍니다.",
        disabled=auto_mode and auto_y is not None
    )
with viz_col3:
    size_col = st.selectbox(
        size_label,
        options=["- 선택 안함 -"] + all_cols,
        index=0 if not auto_size else all_cols.index(auto_size) + 1,
        disabled=auto_mode and auto_size is not None
    )

hover_cols = st.multiselect(
    "💡 차트 툴팁(마우스 오버)에 표시할 추가 정보",
    options=all_cols, default=None, disabled=auto_mode
)
agg_fn = "count"
if chart_type.startswith("막대("):
    agg_fn = st.selectbox("집계 함수(막대)", ["count", "sum", "mean", "median"], help="Y축이 없으면 'count'가 자동 적용됩니다.", disabled=auto_mode and auto_y is None)

def get_val(opt): return None if (opt == "- 선택 안함 -" or opt == "-") else opt
x = x_col if not auto_mode else auto_x or x_col
y = get_val(y_col) if not auto_mode else auto_y or get_val(y_col)
size = get_val(size_col) if not auto_mode else auto_size or get_val(size_col)
hover = hover_cols if hover_cols else None

fig = None; chart_spec = None
try:
    if chart_type.startswith("선("):
        if y is None: st.warning("선 그래프는 Y축이 필요합니다.")
        else:
            fig = px.line(df, x=x, y=y, hover_data=hover, height=500, title=f"{x}에 따른 {y} 변화")
            chart_spec = {"chart_type": "Line", "x": x, "y": y, "hover": hover}
    elif chart_type.startswith("막대("):
        if y is None: 
            tmp = df.groupby(x).size().reset_index(name="count")
            fig = px.bar(tmp, x=x, y="count", hover_data=hover, height=500, title=f"{x}별 개수(count)")
            chart_spec = {"chart_type": "Bar (Count)", "x": x, "y": "count", "hover": hover}
        else: 
            agg_map = {"count": "count", "sum": "sum", "mean": "mean", "median": "median"}
            tmp = df.groupby(x)[y].agg(agg_map[agg_fn]).reset_index()
            y_agg = f"{agg_fn}_{y}"; tmp = tmp.rename(columns={y: y_agg})
            fig = px.bar(tmp, x=x, y=y_agg, hover_data=hover, height=500, title=f"{x}별 {y}의 {agg_fn}")
            chart_spec = {"chart_type": "Bar (Aggregate)", "x": x, "y": y_agg, "function": agg_fn, "hover": hover}
    elif chart_type.startswith("산점도"):
        if y is None: st.warning("산점도는 Y축이 필요합니다.")
        else:
            fig = px.scatter(df, x=x, y=y, size=size, hover_data=hover, opacity=0.7, height=500, title=f"{x}와 {y}의 관계 (크기: {size})")
            chart_spec = {"chart_type": "Scatter", "x": x, "y": y, "size": size, "hover": hover}
    elif chart_type.startswith("원("):
        if y is None: st.warning("원 그래프는 '값 (수치 열)' (Y축)이 필요합니다.")
        else:
            fig = px.pie(df, names=x, values=y, hover_data=hover, height=500, title=f"{x}별 {y}의 비율")
            chart_spec = {"chart_type": "Pie", "names": x, "values": y, "hover": hover}
    elif chart_type.startswith("지도"): 
        if y is None: st.warning("지도 시각화는 '위도'와 '경도' 열이 모두 필요합니다.")
        else:
            fig = px.scatter_geo(df, lat=x, lon=y, size=size, hover_data=hover, projection="natural earth", height=600, title=f"지도 시각화 (위도:{x}, 경도:{y}, 크기:{size})")
            fig.update_geos(center={"lat": 36, "lon": 127.5}, lataxis_range=[33, 39], lonaxis_range=[124, 132], showcountries=True, showcoastlines=True)
            chart_spec = {"chart_type": "Map (Scatter Geo)", "lat": x, "lon": y, "size": size, "hover": hover}
except Exception as e:
    st.error(f"차트 생성 중 오류: {e}")

st.session_state.chart_spec = chart_spec

if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
    if x and y and pd.api.types.is_numeric_dtype(df[y]):
        st.markdown("#### 🔍 간단한 자동 해석")
        st.info(auto_describe_trend(df[[x, y]].dropna(), x, y))
else:
    st.info("위의 옵션을 선택하여 시각화를 생성해 보세요.")


# =========================
# 3) 데이터 해석 챗봇
# =========================
st.markdown("## 3) 데이터 해석 챗봇 🤖")
st.caption("AI에게 데이터와 차트를 분석해 달라고 요청해 보세요.")

# [수정] summarize_dataframe: 통계 요약(describe)을 포함하도록 강화
def summarize_dataframe(df: pd.DataFrame, max_rows: int = 5) -> str:
    """데이터프레임을 AI가 이해하기 쉬운 상세한 JSON 요약으로 변환합니다."""
    
    # 1. 스키마 (데이터 타입) - 열이 많으면 앞 20개만
    limited_cols = df.columns[:20]
    schema = {col: str(df[col].dtype) for col in limited_cols}
    
    # 2. 미리보기 (Head)
    preview = df.head(max_rows).to_dict(orient="records")
    
    # 3. 통계 요약 (Numerical) - 너무 넓을 경우 앞 20개만
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]
    try:
        numerical_summary = df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
    except Exception:
        numerical_summary = {} # 수치형 데이터가 없을 경우
        
    # 4. 범주형 요약 (Categorical) - 앞 20개만
    categorical_summary = {}
    for col in df.select_dtypes(include=['object', 'category']).columns[:20]:
        categorical_summary[col] = {
            "nunique": df[col].nunique(),
            "top_5_values": df[col].value_counts().head(5).to_dict()
        }

    summary = {
        "file_name": file.name if file else "N/A",
        "total_shape": [int(df.shape[0]), int(df.shape[1])],
        "schema": schema,
        "head_preview (5 rows)": preview,
        "numerical_summary (df.describe)": numerical_summary,
        "categorical_summary (top 5 values)": categorical_summary
    }

    # JSON 변환 시 ensure_ascii=False 로 한글 유지
    # indent=2를 넣어 가독성 향상
    return json.dumps(summary, ensure_ascii=False, indent=2, default=str)


def build_messages(prompt, data_brief, chart_spec, add_data_head, add_context):
    # --- RAG ---
    system_prompt = f"""
[역할 & 톤]
너는 중학교 과학 수업에서 장윤하 선생님을 돕는 한국인 과학 보조 교사다. 말투는 친근하고 짧게, 논문체/교사용 안내문처럼 말하지 않는다.

[답변 방식]
- 숫자/경향 해석: 오직 제공된 [데이터 요약], [차트 정보]에 있는 값과 패턴만 사용한다.
- 과학 개념·교육과정 연결: 아래 두 지식 파일의 내용에 기반해 설명한다.
  • [교육과정 지식] (knowledge_curriculum.txt)
  {KNOWLEDGE_CURRICULUM if KNOWLEDGE_CURRICULUM else "N/A"}
  • [과학 원리 지식] (knowledge_disasters.txt)
  {KNOWLEDGE_DISASTERS if KNOWLEDGE_DISASTERS else "N/A"}
- 수업 톤: 학생에게 말하듯 간단히 설명하고, 이어서 “왜 그럴까?” “다른 자료와 비교하면 어떨까?” 같은 생각거리를 1~2개 자연스럽게 던진다.
- 데이터가 부족하면 모르는 부분을 솔직히 말한다.
- 수업 범위를 벗어난 질문엔 “이 챗봇은 중학교 과학 수업 지원용입니다.”라고 답한다.

[출력 형식]
- 짧은 문장, 친근한 구어체 한국어
- 중요한 수치는 근거를 함께 언급
- bullet(•)과 **굵은 글씨**로 핵심을 정리
    """
    
    msgs: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    
    # --- 컨텍스트 ---
    ctx_parts = []
    if add_data_head:
        ctx_parts.append(f"[데이터 요약]\n{data_brief}")
    if add_context and chart_spec:
        ctx_parts.append(f"[현재 시각화된 차트 정보]\n{json.dumps(chart_spec, ensure_ascii=False, indent=2)}")
    
    ctx = "\n\n".join(ctx_parts) if ctx_parts else "(제공된 데이터 컨텍스트 없음)"

    user = f"{prompt}\n\n[참고할 컨텍스트]\n{ctx}"
    msgs.append({"role": "user", "content": user})
    return msgs


# call_openai
def call_openai(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    if not OPENAI_AVAILABLE:
        return "⚠️ openai 패키지를 찾을 수 없습니다. `pip install openai` 후 다시 시도하세요."
    if not api_key or api_key == "YOUR_OPENAI_API_KEY_HERE":
        return "⚠️ OpenAI API Key가 필요합니다. 코드 상단의 `OPENAI_API_KEY` 변수를 수정하세요."
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI 호출 오류: {e}"


# --- 챗봇 UI ---

# 데이터 요약
try:
    data_brief = summarize_dataframe(df, max_rows=5)
except Exception as e:
    data_brief = "데이터 요약이 제공되지 않아, 그래프에서 보이는 정보 중심으로 설명할게."
    st.warning(f"데이터 요약 생성 실패: {e}")

# 프롬프트
default_prompt = (
    "현재 업로드된 [데이터 요약]과 [차트 정보]를 분석해 주세요.\n\n"
    "1. 이 데이터에서 발견할 수 있는 가장 중요한 경향이나 사실은 무엇인가요? (데이터의 숫자를 근거로 들어주세요)\n"
    "2. 이 현상을 [과학 원리 지식]과 어떻게 연결할 수 있나요?\n"
    "3. 이 데이터를 [교육과정 지식]의 성취기준과 연결할 때, 어떤 비판적 질문을 토론해 볼 수 있을까요?"
)
st.markdown("#### 컨텍스트 전달 옵션")
opt_col1, opt_col2 = st.columns([1, 1])
with opt_col1:
    add_context = st.checkbox("그래프 메타데이터 포함", True, help="차트 유형, 축, 집계 방식 등 메타를 LLM에 전달")
with opt_col2:
    add_data_head = st.checkbox("데이터 요약(통계 포함) 포함", True, help="AI가 실제 데이터를 분석하도록 통계 요약본을 전달합니다.")

st.markdown("### 대화")
if st.button("기록 지우기", use_container_width=True):
    st.session_state.chat_history = []
if not st.session_state.chat_history:
    st.info("예시 질문을 눌러 바로 대화를 시작할 수 있어요.")
    if st.button("예시 질문 불러오기", type="secondary"):
        st.session_state.chat_history.append({"role": "user", "content": default_prompt})
        msgs = build_messages(default_prompt, data_brief, st.session_state.chart_spec, add_data_head, add_context)
        answer = call_openai(msgs, st.session_state.model, st.session_state.api_key)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 대화 렌더링
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# 입력창
user_prompt = st.chat_input("질문을 입력하세요")
if user_prompt:
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("데이터를 먼저 업로드해 주세요.")
    elif data_brief.startswith("데이터 요약이 제공되지 않아"):
        st.warning("데이터 요약이 준비되지 않아, 그래프 중심으로만 안내합니다.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            with st.spinner("AI가 데이터를 분석 중입니다..."):
                msgs = build_messages(user_prompt, data_brief, st.session_state.chart_spec, add_data_head, add_context)
                answer = call_openai(msgs, st.session_state.model, st.session_state.api_key)
                st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})



with st.expander("ℹ️ 도움말 / 주의"):
    st.markdown(
        """
- 이 AI 챗봇은 '재해·재난과 안전' 단원 수업을 위해 설정되었습니다.
        """
    )