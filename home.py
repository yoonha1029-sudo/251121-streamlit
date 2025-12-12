import io
import json
import textwrap
from typing import Dict, Any, List, Optional, Tuple
import os
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import requests

# ---- OpenAI 확인 ----
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# =========================
# API 키 로드 함수 (secrets.toml + 환경변수)
# =========================
def load_api_key() -> Optional[str]:
    """
    1순위: .streamlit/secrets.toml 의 OPENAI_API_KEY
    2순위: 환경 변수 OPENAI_API_KEY
    둘 다 없으면 None
    """
    # 1) secrets.toml에서 시도
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    # 2) 환경 변수에서 시도
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    # 3) 실패 시 None
    return None


# =========================
# 지식 파일 로드 헬퍼 (Simplified RAG)
# =========================
@st.cache_data  # 앱 실행 시 한 번만 읽도록 캐시
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
    page_title="AI 기반 빅데이터 탐구",
    page_icon="images/extreme.png",
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
    # 여기서만 api_key 초기화 (OPENAI_API_KEY 상수 사용 X)
    st.session_state.api_key = load_api_key() or ""
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o-mini"
if "chart_spec" not in st.session_state:
    st.session_state.chart_spec: Optional[Dict[str, Any]] = None


# =========================
# 사이드바: AI 모델 설정
# =========================
with st.sidebar:
    st.markdown("## AI 모델 설정")

    if not st.session_state.api_key:
        st.error(
            "OpenAI API 키를 찾을 수 없습니다.\n\n"
            "아래 중 한 가지 방법으로 설정해 주세요.\n"
            "1) 프로젝트 폴더 안에 `.streamlit/secrets.toml` 생성 후\n"
            '   `OPENAI_API_KEY = "실제_키"` 입력\n'
            "2) 환경 변수 OPENAI_API_KEY 설정"
        )
    else:
        st.success("OpenAI API Key가 로드되었습니다.")

    st.session_state.model = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0,
        help="해석 정확도가 중요하면 상위 모델, 비용이 중요하면 mini 권장",
    )
    st.divider()
    st.info("먼저 데이터를 왼쪽 데이터 자료실 페이지에서 준비해 주세요.")

# =========================
# 상단 헤더
# =========================
st.title("재해·재난과 안전 빅데이터 탐구하기")
st.markdown(
    "‘재해·재난과 안전’ **빅데이터 탐구 수업**을 돕는 웹사이트입니다. "
    "데이터를 시각화하고, **AI에게 해석**을 요청해 보세요."
)

# API 키 없으면 여기서 바로 중단
if not st.session_state.api_key:
    st.stop()


# =========================
# 1) 데이터 불러오기
# =========================
st.markdown("## 데이터 불러오기")
file = st.file_uploader(
    "CSV 또는 XLSX 파일 업로드",
    type=["csv", "xlsx"],
    accept_multiple_files=False,
    help="첫 번째 시트 기준(XLSX). 수업용 데이터는 'data' 페이지에서 다운로드 받으세요.",
)


def load_dataframe(_file) -> pd.DataFrame:
    if _file is None:
        return pd.DataFrame()
    if _file.name.lower().endswith(".csv"):
        try:
            df = pd.read_csv(_file, sep=",", low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(_file, sep=",", low_memory=False, encoding='cp949')
    else:
        df = pd.read_excel(_file, engine="openpyxl")
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

EARTHQUAKE_LAT_KEYWORDS = ["위도", "latitude", "lat"]
EARTHQUAKE_LON_KEYWORDS = ["경도", "longitude", "lon"]
EARTHQUAKE_MAG_KEYWORDS = ["규모", "진도", "magnitude", "mag"]


def detect_earthquake_columns(df: pd.DataFrame):
    """
    지진 데이터로 보이면 (위도, 경도, 규모/진도) 컬럼을 찾아서 돌려줌.
    최소한 위도+경도 두 개만 있으면 '지도' 자동 추천.
    """
    lat_col, lon_col, mag_col = None, None, None

    for col in df.columns:
        lower = col.lower()
        if any(k in lower for k in EARTHQUAKE_LAT_KEYWORDS):
            lat_col = col
        if any(k in lower for k in EARTHQUAKE_LON_KEYWORDS):
            lon_col = col
        if any(k in lower for k in EARTHQUAKE_MAG_KEYWORDS):
            mag_col = mag_col or col  # 여러 개면 첫 번째만

    if lat_col and lon_col:
        return lat_col, lon_col, mag_col

    return None, None, None

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

    lat_col, lon_col, mag_col = detect_earthquake_columns(df)
    if lat_col and lon_col:
        return lat_col, lon_col, mag_col, "지도 (위도/경도)"
    
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
    - x 기준으로 정렬
    - 처음 1/3 vs 마지막 1/3 평균을 비교해서 증가/감소 판단
    - 변화량이 전체 범위에 비해 작으면 '뚜렷한 경향 없음' 처리
    """
    if x not in df.columns or y not in df.columns:
        return ""

    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        return ""

    # x가 숫자나 날짜면 x 기준으로 정렬
    if pd.api.types.is_numeric_dtype(tmp[x]) or pd.api.types.is_datetime64_any_dtype(tmp[x]):
        tmp = tmp.sort_values(by=x)

    series = tmp[y]
    if not pd.api.types.is_numeric_dtype(series):
        return ""

    n = len(series)
    if n < 3:
        return ""

    # 처음/마지막 1/3 평균 비교
    k = max(1, n // 3)
    first_mean = series.iloc[:k].mean()
    last_mean = series.iloc[-k:].mean()
    diff_mean = last_mean - first_mean

    data_min, data_max = series.min(), series.max()
    data_range = data_max - data_min if data_max != data_min else 0

    # 변화량이 너무 작으면 뚜렷한 경향 없음
    # (전체 범위의 10% 미만 변화는 '크게 변하지 않는다'로 처리)
    if data_range == 0:
        trend_desc = "전체 값의 크기가 거의 일정합니다."
        direction_flag = 0
    else:
        rel_change = abs(diff_mean) / data_range
        if rel_change < 0.1:
            trend_desc = "전체적으로 큰 증가나 감소 없이 비슷한 수준을 유지합니다."
            direction_flag = 0
        elif diff_mean > 0:
            trend_desc = "전체적으로 시간이 지날수록 값이 감소하기보다는 **늘어나는 경향**이 있습니다."
            direction_flag = 1
        else:
            trend_desc = "전체적으로 시간이 지날수록 값이 증가하기보다는 **줄어드는 경향**이 있습니다."
            direction_flag = -1

    # 변동성(오르내림) 체크
    diffs = series.diff().dropna()
    if not diffs.empty:
        up_ratio = (diffs > 0).mean()
        down_ratio = (diffs < 0).mean()
    else:
        up_ratio = down_ratio = 0.0

    if up_ratio > 0.6:
        var_desc = "중간에 조금씩 내려갈 때도 있지만, 전반적으로는 올라가는 구간이 더 많습니다."
    elif down_ratio > 0.6:
        var_desc = "중간에 조금씩 오를 때도 있지만, 전반적으로는 내려가는 구간이 더 많습니다."
    else:
        var_desc = "값이 오르내림을 반복하여 변동이 꽤 있는 편입니다."

    # 대표 수치 한 줄
    summary_text = f"처음 구간 평균은 약 {first_mean:.2f}, 마지막 구간 평균은 약 {last_mean:.2f}입니다."

    # 너무 애매하면 경향 문장을 부드럽게
    if direction_flag == 0:
        return f"{trend_desc} {var_desc} {summary_text}"
    else:
        return f"{trend_desc} {var_desc} {summary_text}"



# --- 파일 업로드 처리 ---
if file:
    df = load_dataframe(file)
    df = optimize_dtypes(df)
    st.session_state.df = df

if st.session_state.df is not None and not st.session_state.df.empty:
    df = st.session_state.df
    st.success(f"불러온 데이터: {df.shape[0]:,}행 × {df.shape[1]:,}열")

    # --- 미리보기 ---
    with st.expander("📋 데이터 미리보기 (상위 100행)", expanded=True):
        st.dataframe(df.head(100), use_container_width=True)

    # --- 간단 요약 ---
    st.markdown("### 📊 데이터 요약")
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.metric("행 수", f"{df.shape[0]:,}")
    with col_meta2:
        st.metric("열 수", f"{df.shape[1]:,}")

    # 결측치 존재 여부만 표시 (숫자 없음)
    if df.isna().sum().sum() > 0:
        st.warning("⚠️ 일부 열에 결측치가 있습니다. (그래프에는 큰 문제 없음)")
    else:
        st.success("결측치 없음!")

else:
    st.info("왼쪽 사이드바에서 **데이터 자료실** 페이지를 클릭해 분석할 데이터를 준비하세요.")
    st.stop()


# =========================
# 2) 데이터 시각화
# =========================
st.markdown("## 데이터 시각화")
st.caption("핵심 차트 유형만 선택하고, AI와 함께 해석에 집중해 보세요.")
auto_mode = st.checkbox(
    "🔀 자동 차트 추천 사용",
    value=True,
    help="데이터에서 시간/연도/주차/수치 열을 찾아 자동으로 차트를 만듭니다."
)

all_cols = df.columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ✅ 공통 차트 라벨 리스트
CHART_LABELS = ["선(line)", "막대(bar)", "산점도(scatter)", "원(pie)", "지도 (위도/경도)"]

# 자동 추천 실행 (size는 더 이상 쓰지 않으므로 _로 버림)
auto_x, auto_y, _, auto_chart_label = infer_chart(df)

if auto_mode:
    st.info(
        f"추천 결과: 차트 유형='{auto_chart_label}', X축='{auto_x}', "
        f"Y축='{auto_y if auto_y else '없음'}'"
    )
    if auto_y is None and auto_chart_label != "원(pie)":
        st.warning("수치형 열을 찾지 못했습니다. 필요하면 자동 모드를 끄고 직접 Y축을 선택하세요.")

# 차트 유형 선택
chart_type = st.selectbox(
    "차트 유형",
    CHART_LABELS,
    index=CHART_LABELS.index(auto_chart_label) if (auto_mode and auto_chart_label in CHART_LABELS) else 0,
    disabled=auto_mode
)

# 축 레이블 설정 (size 관련 전부 제거)
if chart_type.startswith("원("):
    x_label = "이름 (범주 열)"
    y_label = "값 (수치 열)"
elif chart_type.startswith("지도"):
    x_label = "위도 (Latitude) 열"
    y_label = "경도 (Longitude) 열"
else:
    x_label = "X축"
    y_label = "Y축 (필요시)"

# 축 선택 (x, y만)
viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    x_col = st.selectbox(
        x_label,
        options=all_cols,
        index=all_cols.index(auto_x) if auto_mode and auto_x in all_cols else 0,
        disabled=auto_mode and auto_x is not None
    )
with viz_col2:
    # 지도/산점도/선/막대 등에서 Y축 선택
    y_options = ["- 선택 안함 -"] + (numeric_cols if numeric_cols else all_cols)
    y_default = auto_y if auto_mode and auto_y in y_options else "- 선택 안함 -"
    # 원 그래프는 values가 필수라서 수치형 우선
    if chart_type.startswith("원("):
        y_options_pie = numeric_cols if numeric_cols else all_cols
        y_default_pie = auto_y if auto_mode and auto_y in y_options_pie else (
            y_options_pie[0] if y_options_pie else None
        )
        y_col = st.selectbox(
            y_label,
            options=y_options_pie,
            index=y_options_pie.index(y_default_pie) if (y_default_pie and y_default_pie in y_options_pie) else 0
        )
    else:
        y_col = st.selectbox(
            y_label,
            options=y_options,
            index=y_options.index(y_default) if y_default in y_options else 0,
            help="수치형 열을 우선 보여줍니다.",
            disabled=auto_mode and auto_y is not None
        )

# 툴팁용 컬럼
hover_cols = st.multiselect(
    "💡 차트 툴팁(마우스 오버)에 표시할 추가 정보",
    options=all_cols,
    default=None,
    disabled=auto_mode
)

# 막대 그래프 집계 함수
agg_fn = "count"
if chart_type.startswith("막대("):
    agg_fn = st.selectbox(
        "집계 함수(막대)",
        ["count", "sum", "mean", "median"],
        help="Y축이 없으면 'count'가 자동 적용됩니다.",
        disabled=auto_mode and auto_y is None
    )


def get_val(opt: str):
    return None if (opt == "- 선택 안함 -" or opt == "-") else opt


# 실제 축 값 결정
x = x_col if not auto_mode else auto_x or x_col
if chart_type.startswith("원("):
    y = y_col  # 파이 차트는 y 필수
else:
    y = get_val(y_col) if not auto_mode else auto_y or get_val(y_col)

hover = hover_cols if hover_cols else None

fig = None
chart_spec = None

try:
    # 1) 선 그래프
    if chart_type.startswith("선("):
        if y is None:
            st.warning("선 그래프는 Y축이 필요합니다.")
        else:
            fig = px.line(
                df,
                x=x,
                y=y,
                hover_data=hover,
                height=500,
                title=f"{x}에 따른 {y} 변화"
            )
            chart_spec = {"chart_type": "Line", "x": x, "y": y, "hover": hover}

    # 2) 막대 그래프
    elif chart_type.startswith("막대("):
        if y is None:
            # x별 개수
            tmp = df.groupby(x).size().reset_index(name="count")
            fig = px.bar(
                tmp,
                x=x,
                y="count",
                hover_data=hover,
                height=500,
                title=f"{x}별 개수(count)"
            )
            chart_spec = {"chart_type": "Bar (Count)", "x": x, "y": "count", "hover": hover}
        else:
            agg_map = {"count": "count", "sum": "sum", "mean": "mean", "median": "median"}
            tmp = df.groupby(x)[y].agg(agg_map[agg_fn]).reset_index()
            y_agg = f"{agg_fn}_{y}"
            tmp = tmp.rename(columns={y: y_agg})
            fig = px.bar(
                tmp,
                x=x,
                y=y_agg,
                hover_data=hover,
                height=500,
                title=f"{x}별 {y}의 {agg_fn}"
            )
            chart_spec = {
                "chart_type": "Bar (Aggregate)",
                "x": x,
                "y": y_agg,
                "function": agg_fn,
                "hover": hover
            }

    # 3) 산점도
    elif chart_type.startswith("산점도"):
        if y is None:
            st.warning("산점도는 Y축이 필요합니다.")
        else:
            fig = px.scatter(
                df,
                x=x,
                y=y,
                hover_data=hover,
                opacity=0.7,
                height=500,
                title=f"{x}와 {y}의 관계"
            )
            chart_spec = {"chart_type": "Scatter", "x": x, "y": y, "hover": hover}

    # 4) 원 그래프
    elif chart_type.startswith("원("):
        if y is None:
            st.warning("원 그래프는 '값 (수치 열)'이 필요합니다.")
        else:
            fig = px.pie(
                df,
                names=x,
                values=y,
                hover_data=hover,
                height=500,
                title=f"{x}별 {y}의 비율"
            )
            chart_spec = {"chart_type": "Pie", "names": x, "values": y, "hover": hover}

    # 5) 지도 (지진 데이터)
    elif chart_type.startswith("지도"):
        if y is None:
            st.warning("지도 시각화는 '위도'와 '경도' 열이 모두 필요합니다.")
        else:
            fig = px.scatter_geo(
                df,
                lat=x,
                lon=y,
                hover_data=hover,
                projection="natural earth",
                height=600,
                title=f"지도 시각화 (위도:{x}, 경도:{y})"
            )
            fig.update_geos(
                center={"lat": 36, "lon": 127.5},
                lataxis_range=[33, 39],
                lonaxis_range=[124, 132],
                showcountries=True,
                showcoastlines=True,
            )
            chart_spec = {
                "chart_type": "Map (Scatter Geo)",
                "lat": x,
                "lon": y,
                "hover": hover
            }

except Exception as e:
    st.error(f"차트 생성 중 오류: {e}")

st.session_state.chart_spec = chart_spec

if fig is not None:
    st.plotly_chart(fig, use_container_width=True)
    # 자동 해석은 숫자 y축 있을 때만
    if x and y and (y in df.columns) and pd.api.types.is_numeric_dtype(df[y]):
        st.markdown("#### 간단한 자동 해석")
        st.info(auto_describe_trend(df[[x, y]].dropna(), x, y))
else:
    st.info("위의 옵션을 선택하여 시각화를 생성해 보세요.")


# =========================
# 3) 데이터 해석 챗봇
# =========================
st.markdown("## 데이터 해석 챗봇")
st.caption("그래프를 보고 궁금한 점을 챗봇에게 물어보세요.")


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
        numerical_summary = {}

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


def call_openai(messages: List[Dict[str, str]], model: str, api_key: str) -> str:
    if not OPENAI_AVAILABLE:
        return "⚠️ openai 패키지를 찾을 수 없습니다. `pip install openai` 후 다시 시도하세요."
    if not api_key:
        return "⚠️ OpenAI API Key가 필요합니다. .streamlit/secrets.toml 또는 환경 변수 OPENAI_API_KEY를 설정해 주세요."

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
        # 여기서 "Connection error"도 포함해서 다 보여줌
        return f"❌ OpenAI 호출 오류: {type(e).__name__}: {e}"


# --- 챗봇 UI ---

# 1. 데이터 요약 로직
try:
    data_brief = summarize_dataframe(df, max_rows=5)
except Exception as e:
    data_brief = "데이터 요약이 제공되지 않아, 그래프에서 보이는 정보 중심으로 설명할게."
    st.warning(f"데이터 요약 생성 실패: {e}")

# 2. 헤더 및 설정
col_head, col_opt = st.columns([3, 1])
with col_head:
    st.subheader("AI 데이터 탐구 대화")

with st.popover("⚙️ 대화 설정"):
    st.caption("AI에게 어느 정도 정보를 넘겨줄지 정하는 곳이에요. 보통은 기본값 그대로 두면 됩니다.")
    add_context = st.checkbox("현재 그래프 정보도 같이 알려주기", True)
    add_data_head = st.checkbox("데이터 표 일부(요약)도 같이 알려주기", True)

    st.divider()
    st.caption("❗ 문제가 생겼을 때만 아래 버튼을 눌러요.")
    if st.button("🧺 대화 기록 초기화", type="primary", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 3. 대화 내용 렌더링
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("데이터에 대해 궁금한 점을 직접 입력하거나, 아래 예시 버튼을 눌러보세요.")

        btn_col1, btn_col2, btn_col3 = st.columns(3)
        selected_prompt = None

        with btn_col1:
            if st.button("데이터 경향 분석", use_container_width=True):
                selected_prompt = (
                    "현재 데이터에서 발견할 수 있는 가장 중요한 경향을 숫자를 들어 설명해 줘."
                )
        with btn_col2:
            if st.button("과학 원리 연결", use_container_width=True):
                selected_prompt = (
                    "이 데이터에 나타난 현상을 교과서에 나오는 과학 원리와 연결해서 설명해 줘."
                )
        with btn_col3:
            if st.button("심화 탐구(기상)", use_container_width=True):
                selected_prompt = (
                    "기상 데이터(기온, 강수량 등)와 재해 발생의 연관성을 분석하고, 추가로 탐구해볼 주제를 추천해 줘."
                )

        if selected_prompt:
            st.session_state.chat_history.append({"role": "user", "content": selected_prompt})

            msgs = build_messages(selected_prompt, data_brief, st.session_state.chart_spec, add_data_head, add_context)
            answer = call_openai(msgs, st.session_state.model, st.session_state.api_key)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

# 4. 입력창 및 응답 처리
if user_prompt := st.chat_input("그래프를 보며 궁금한 점을 적어 보세요. (예: 최근 10년 동안 어떻게 변했어?)"):
    if st.session_state.df is None or st.session_state.df.empty:
        st.warning("데이터를 먼저 업로드해 주세요.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("분석 중입니다..."):
                if data_brief.startswith("데이터 요약이 제공되지 않아"):
                    st.caption("참고: 데이터 요약 없이 차트 정보만으로 분석합니다.")

                msgs = build_messages(user_prompt, data_brief, st.session_state.chart_spec, add_data_head, add_context)
                answer = call_openai(msgs, st.session_state.model, st.session_state.api_key)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# 5. 하단 도움말
with st.expander("힌트! 어떤 질문을 하면 좋을까?"):
    st.markdown(
        """
        ### 추천 프롬프트 예시

        **① 데이터 이해가 안되면**  
        * `이 데이터에서 가장 기본적으로 알 수 있는 내용을 쉽게 설명해 줘.`  
        * `열 이름이 너무 낯설어. 각 열이 무엇을 뜻하는지 중학생 눈높이로 정리해 줘.`  
        * `최근 5년(또는 5개 구간) 동안 값이 어떻게 변했는지 간단히 요약해 줘.`  

        **② 그래프 읽기가 여려우면**  
        * `지금 차트에서 가장 눈에 띄는 증가 또는 감소 구간을 알려줘.`  
        * `이 데이터에서 최고값·최저값이 언제(어디서) 나타났는지 알려주고, 그 이유를 추측해줘.`  
        * `두 변수의 관계(예: 기온과 재해 건수)를 그래프를 보며 설명해 줘.`  

        **③ 재해·재난 연결이 어려우면**  
        * `이 데이터가 재해·재난과 어떤 관련이 있는지, 실제 사례를 들어 설명해 줘.`  
        * `데이터를 보면 안전을 위해 어떤 준비가 필요해 보이는지 정리해 줘.`  
        * `비슷한 데이터를 더 모은다면 어떤 걸 조사해 보면 좋을지, 추가 데이터 아이디어를 3개만 제안해 줘.`  

        **④ 기후 변화 & 심화 탐구를 해 보고 싶으면**  
        * `이 데이터가 기후 변화와 관련되어 있다면, 어떤 점에서 연결된다고 볼 수 있을까? 근거를 들어 설명해 줘.`  
        * `기후 변화로 앞으로 이런 재해가 어떻게 바뀔지, 데이터를 바탕으로 합리적인 예측을 해 줘. (너무 단정적으로 말하지 말고 가능성 위주로 말해줘.)`  
        * `이 데이터를 가지고 "기후 위기와 안전"을 주제로 친구들과 토론한다면, 던져볼 만한 토론 질문을 3~4개 만들어 줘.`  
        * `이 데이터를 이용해서 3분 발표를 한다고 할 때, 발표 개요(도입–본론–결론)를 짜 줘.`  
        """
    )

with st.expander("웹사이트 사용 소감 남기기"):
    st.markdown(
        """
        이 웹앱을 쓰면서 느낀 점이나, 개선했으면 하는 점이 있다면 여기서 남겨 주세요.  
        선생님이 다음 수업을 더 좋게 만드는 데 큰 도움이 됩니다. 🙂
        
        👉 **구글 설문 전체 화면에서 작성하고 싶다면** 아래 링크를 눌러도 돼요.
        """
    )
    st.markdown("[Google Form 바로가기](https://forms.gle/fx7WyUL78gkQ2t8PA)")

    # --- Google Form 설정 ---
    GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdyo9JuRoTCH_QsSKghM_AE9Pwz0vC0yJyPL4zxc_yD68A61A/formResponse"

    # 각 문항에 해당하는 entry 번호 (실제 번호로 교체 완료)
    ENTRY_NAME = "entry.693418327"        # 학번과 이름
    ENTRY_RESEARCH = "entry.1589337783"   # 내가 탐구한 재해/재난과 탐구 내용
    ENTRY_FEEDBACK = "entry.786544321"    # 웹사이트 사용 소감 및 선생님께 하고 싶은 말

    st.markdown("---")

    st.markdown("⬇️ 아래에 바로 입력하면, 내용이 **Google Form 스프레드시트에 자동 저장**됩니다.")

    with st.form("feedback_form"):
        name = st.text_input("학번과 이름")
        msg_research = st.text_area("내가 탐구한 재해/재난과 탐구 내용을 적어주세요. (2-3문장)")
        msg_feedback = st.text_area(
            "웹사이트 사용 소감, 개선하면 좋을 점, 또는 장윤하 쌤에게 하고 싶은 말을 자유롭게 적어주세요. ^_^"
        )

        # SSL 우회 옵션 (학교/기관망에서 인증서 에러 날 때만)
        ignore_ssl = st.checkbox(
            "SSL 인증서 검증 무시하고 전송하기 (학교/기관 네트워크에서 오류가 날 때만 체크)",
            value=False
        )

        submitted = st.form_submit_button("제출")

    if submitted:
        if not msg_research.strip() and not msg_feedback.strip():
            st.warning("내용을 한 줄 이상 적어 주세요.")
        else:
            data = {
                ENTRY_NAME: name,
                ENTRY_RESEARCH: msg_research,
                ENTRY_FEEDBACK: msg_feedback,
            }

            try:
                if ignore_ssl:
                    # ⚠️ 보안상 완전히 안전한 방법은 아니라서, 학교/기관 내부망에서만 사용하는 게 좋아요.
                    response = requests.post(GOOGLE_FORM_URL, data=data, timeout=10, verify=False)
                else:
                    response = requests.post(GOOGLE_FORM_URL, data=data, timeout=10)

                # Google Form은 보통 200 또는 302(리다이렉트)를 돌려줌
                if response.status_code in (200, 302):
                    st.success("피드백이 성공적으로 제출되었습니다! 🙌")
                else:
                    st.warning(f"요청은 전송했지만, 응답 코드가 예상과 다릅니다: {response.status_code}")
            except requests.exceptions.SSLError as e:
                st.error(
                    "SSL 인증서 오류가 발생했어요. 학교/기관 네트워크에서 HTTPS를 중간에서 검사할 때 자주 생기는 문제입니다.\n"
                    "위의 'SSL 인증서 검증 무시하고 전송하기' 체크를 활성화한 뒤 다시 제출해 보세요."
                )
                st.code(str(e))
            except Exception as e:
                st.error(f"제출 중 오류가 발생했습니다: {e}")