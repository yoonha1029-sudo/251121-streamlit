import streamlit as st

@st.cache_data
def load_local_file_bytes(file_path: str):
    """로컬 파일을 바이트(bytes)로 읽어옵니다."""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return None

# =========================
# 페이지 구성
# =========================
st.title("💾 수업용 데이터 자료실")
st.caption("주제를 선택하여 수업용 CSV 파일을 다운로드하거나 원본 출처를 확인하세요.")
st.info("이 페이지의 파일들은 교사가 수업용으로 미리 정제한 데이터입니다. 원본 데이터는 각 기관의 공개 자료를 바탕으로 합니다.")

st.markdown("---")

# =========================
# 1. 국내 기상·기후 데이터
# =========================
st.subheader("국내 기상·기후 데이터")

col1, col2 = st.columns(2)

with col1:
    # 국내 평균 기온
    f_temp = load_local_file_bytes("국내_기온_데이터.csv")
    st.markdown("### 🌡️ 국내 평균기온 데이터")
    st.write("연도별 평균기온 변화를 볼 수 있는 데이터입니다.")
    if f_temp:
        st.download_button(
            label="CSV 다운로드",
            data=f_temp,
            file_name="국내_기온_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: [기상자료개방포털](https://data.kma.go.kr)\n"
    )

    st.markdown("---")

    # 국내 서리일수
    f_frost = load_local_file_bytes("국내_서리일수_데이터.csv")
    st.markdown("### 🧊 국내 서리일수 데이터")
    st.write("연도별 서리 발생 일수를 정리한 데이터입니다.")
    if f_frost:
        st.download_button(
            label="CSV 다운로드",
            data=f_frost,
            file_name="국내_서리일수_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: [기상자료개방포털](https://data.kma.go.kr)\n"
    )

with col2:
    # 국내 열대야
    f_tropical_night = load_local_file_bytes("국내_열대야일수_데이터.csv")
    st.markdown("### 🌃 국내 열대야일수 데이터")
    st.write("연도별 열대야(최저기온 25℃ 이상) 발생 일수 데이터입니다.")
    if f_tropical_night:
        st.download_button(
            label="CSV 다운로드",
            data=f_tropical_night,
            file_name="국내_열대야일수_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: [기상자료개방포털](https://data.kma.go.kr)\n"
    )

    st.markdown("---")

    # 국내 인플루엔자
    f_flu = load_local_file_bytes("국내_인플루엔자_데이터.csv")
    st.markdown("### 🤧 국내 인플루엔자(독감) 지표 데이터")
    st.write("연도 또는 주차별 인플루엔자 의사환자 지표를 정리한 데이터입니다.")
    if f_flu:
        st.download_button(
            label="CSV 다운로드",
            data=f_flu,
            file_name="국내_인플루엔자_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: 질병관리청 감염병 포털, 감염병 감시체계 자료\n"
    )

st.markdown("---")

# =========================
# 2. 전 세계 재해·환경 데이터
# =========================
st.subheader("🌍 전 세계 재해·환경 데이터")

# 2-1. 홍수, 지진
col3, col4 = st.columns(2)

with col3:
    # 세계 기록적 홍수
    f_flood = load_local_file_bytes("세계_기록적홍수_데이터.csv")
    st.markdown("### 🌊 세계 기록적 홍수 데이터")
    st.write("1985년 이후 전 세계에서 보고된 대규모·극심한 홍수 사건 수를 정리한 데이터입니다.")
    if f_flood:
        st.download_button(
            label="CSV 다운로드",
            data=f_flood,
            file_name="세계_기록적홍수_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: Dartmouth Flood Observatory\n"
        "[http://floodobservatory.colorado.edu/Archives/index.html](http://floodobservatory.colorado.edu/Archives/index.html)"
    )

    st.markdown("---")

    # 세계 산불(산림 손실)
    f_forest_fire = load_local_file_bytes("세계_산불_데이터.csv")
    st.markdown("### 🔥 세계 산불·산림 손실 데이터")
    st.write("연도별 산림 손실 면적, 산불로 인한 산림 손실 면적 등을 정리한 데이터입니다.")
    if f_forest_fire:
        st.download_button(
            label="CSV 다운로드",
            data=f_forest_fire,
            file_name="세계_산불_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: [Global Forest Watch](https://www.globalforestwatch.org/dashboards/global/?category=land-cover&location=WyJnbG9iYWwiXQ%3D%3D)\n"
    )

with col4:
    # 세계 이산화탄소
    f_carbon_emission = load_local_file_bytes("세계_연이산화탄소배출량_데이터.csv")
    st.markdown("### 🟢 세계 이산화탄소(CO₂) 농도/배출 데이터")
    st.write("연도별 대기 중 이산화탄소 농도 또는 CO₂ 배출량 데이터를 정리한 자료입니다.")
    if f_carbon_emission:
        st.download_button(
            label="CSV 다운로드",
            data=f_carbon_emission,
            file_name="세계_연이산화탄소배출량_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **원본 출처**: [NOAA CO₂ 데이터](https://gml.noaa.gov/ccgg/trends/gl_data.html)\n"

    st.markdown("---")

    # 세계 지진
    f_earthquake = load_local_file_bytes("세계_지진_진도6이상_데이터.csv")
    st.markdown("### 🌏 세계 지진(규모 6 이상) 데이터")
    st.write("1900년 이후 규모 6.0 이상 지진의 연도별 발생 건수 등을 정리한 데이터입니다.")
    if f_earthquake:
        st.download_button(
            label="CSV 다운로드",
            data=f_earthquake,
            file_name="세계_지진_진도6이상_데이터.csv",
            mime="text/csv",
        )
    st.markdown(
        "- **출처**: USGS Earthquake Catalog\n"
        "[https://www.usgs.gov/programs/earthquake-hazards/lists-maps-and-statistics](https://www.usgs.gov/programs/earthquake-hazards/lists-maps-and-statistics)"
    )

st.markdown("---")

# =========================
# 3. 기타 참고 사이트 안내
# =========================
with st.expander("🔗 추가로 참고할 수 있는 공신력 있는 데이터 포털 보기"):
    st.markdown(
        """
- **KOSIS 국가통계포털**: [https://kosis.kr](https://kosis.kr)  
  - 인구, 보건, 환경, 재해 관련 국내 통계
- **기상자료개방포털**: [https://data.kma.go.kr](https://data.kma.go.kr)  
  - 기온, 강수량, 폭염·열대야, 기후변화 관련 기상 자료
- **Global Forest Watch**: [https://www.globalforestwatch.org](https://www.globalforestwatch.org)  
  - 전 세계 산림 손실, 산불, 토지피복 변화 데이터
- **NOAA GML CO₂ 데이터**: [https://gml.noaa.gov/ccgg/trends/](https://gml.noaa.gov/ccgg/trends/)  
  - 대기 중 CO₂ 농도, 장기 추세
- **USGS Earthquake Hazards Program**: [https://earthquake.usgs.gov](https://earthquake.usgs.gov)  
  - 전 세계 지진 목록, 규모·위치·깊이 정보
"""
    )