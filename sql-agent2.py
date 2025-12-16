import os
import streamlit as st
import tempfile
import sqlite3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
import logging
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안 함

# Plotly는 선택적 (없어도 작동)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly가 설치되어 있지 않습니다. matplotlib만 사용합니다.")

# 환경 변수 로드
load_dotenv()

# 로깅 설정
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_filename = os.path.join(log_dir, f"sql_agent_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# HTTP 요청 로그 비활성화
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_openai").setLevel(logging.WARNING)

# 구분선 및 취소선 제거 함수
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    # 취소선 마크다운 제거 (~~텍스트~~ -> 텍스트)
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    # 여러 줄에 걸친 구분선 제거 (공백 포함)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    # 단독 라인의 구분선 제거
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    # 연속된 빈 줄 정리 (최대 2개)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# LLM 모델 선택 함수
def get_llm(model_name: str, temperature: float = 0.7):
    """선택된 모델명에 따라 적절한 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)
    elif model_name == "claude-sonnet-4-5":
        return ChatAnthropic(model="claude-sonnet-4-5", temperature=temperature)
    elif model_name == "gemini-3-pro-preview":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY가 환경변수에 설정되어 있지 않습니다.")
            st.stop()
        return ChatGoogleGenerativeAI(model="gemini-3-pro-preview", google_api_key=api_key, temperature=temperature)
    else:
        # 기본값: gpt-5.1
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)

# 모든 열려있는 데이터베이스 연결 닫기
def close_all_databases():
    """모든 열려있는 SQLite 데이터베이스 연결을 닫습니다."""
    if "db_connections" in st.session_state:
        for conn in st.session_state.db_connections:
            try:
                conn.close()
            except:
                pass
        st.session_state.db_connections = []
    
    if "sql_db" in st.session_state:
        try:
            # SQLDatabase 객체는 내부적으로 연결을 관리하므로 명시적으로 닫을 수 없음
            # 대신 None으로 설정
            st.session_state.sql_db = None
        except:
            pass

# 파일명에서 테이블명 생성
def sanitize_table_name(filename):
    """파일명에서 유효한 테이블명을 생성합니다."""
    # 확장자 제거
    table_name = os.path.splitext(os.path.basename(filename))[0]
    # 테이블명에 특수문자가 있으면 언더스코어로 변경
    table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
    # 숫자로 시작하면 앞에 언더스코어 추가
    if table_name and table_name[0].isdigit():
        table_name = '_' + table_name
    # 빈 문자열이면 기본값 사용
    if not table_name:
        table_name = 'table'
    return table_name

# 파일을 DataFrame으로 읽기
def read_file_to_df(file):
    """CSV 또는 엑셀 파일을 DataFrame으로 읽습니다."""
    file_ext = os.path.splitext(file.name)[1].lower()
    
    if file_ext == '.csv':
        # CSV 파일 읽기 (인코딩 자동 감지)
        try:
            # BytesIO로 변환
            file.seek(0)
            df = pd.read_csv(BytesIO(file.read()), encoding='utf-8')
        except UnicodeDecodeError:
            try:
                file.seek(0)
                df = pd.read_csv(BytesIO(file.read()), encoding='cp949')
            except:
                file.seek(0)
                df = pd.read_csv(BytesIO(file.read()), encoding='latin-1')
    elif file_ext in ['.xlsx', '.xls']:
        # 엑셀 파일 읽기
        file.seek(0)
        try:
            # openpyxl 엔진 사용 (.xlsx)
            if file_ext == '.xlsx':
                df = pd.read_excel(BytesIO(file.read()), engine='openpyxl')
            else:
                # xlrd 엔진 사용 (.xls)
                df = pd.read_excel(BytesIO(file.read()), engine='xlrd')
        except Exception as e:
            # 엔진이 없으면 다른 방법 시도
            try:
                file.seek(0)
                df = pd.read_excel(BytesIO(file.read()))
            except Exception as e2:
                raise Exception(f"엑셀 파일 읽기 실패: {e2}. openpyxl 또는 xlrd 라이브러리가 필요할 수 있습니다.")
    else:
        raise Exception(f"지원하지 않는 파일 형식: {file_ext}")
    
    return df

# 여러 파일을 SQLite DB로 변환
def files_to_db(uploaded_files, db_name):
    """여러 CSV/엑셀 파일을 하나의 SQLite 데이터베이스로 변환합니다."""
    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, db_name)
        
        conn = sqlite3.connect(db_path)
        table_names = []
        
        # 각 파일을 처리
        for uploaded_file in uploaded_files:
            try:
                # 파일 읽기
                df = read_file_to_df(uploaded_file)
                
                # 테이블명 생성
                table_name = sanitize_table_name(uploaded_file.name)
                
                # 같은 이름의 테이블이 있으면 번호 추가
                original_table_name = table_name
                counter = 1
                while table_name in table_names:
                    table_name = f"{original_table_name}_{counter}"
                    counter += 1
                
                # 데이터베이스에 저장
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                table_names.append(table_name)
                
                logger.info(f"파일 '{uploaded_file.name}' -> 테이블 '{table_name}' 변환 완료 ({len(df)}행)")
                
            except Exception as e:
                logger.error(f"파일 '{uploaded_file.name}' 처리 중 오류: {e}")
                st.warning(f"파일 '{uploaded_file.name}' 처리 중 오류가 발생했습니다: {str(e)}")
                continue
        
        conn.close()
        
        if not table_names:
            raise Exception("처리된 파일이 없습니다.")
        
        return db_path, table_names
    except Exception as e:
        logger.error(f"파일들을 DB로 변환하는 중 오류: {e}")
        raise e

# 데이터베이스 정보 가져오기
def get_db_info(db_path):
    """데이터베이스의 테이블, 컬럼, row 수 정보를 가져옵니다."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 테이블 목록 가져오기
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    db_info = {}
    for table in tables:
        # 컬럼 정보 가져오기
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Row 수 가져오기
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row_count = cursor.fetchone()[0]
        
        db_info[table] = {
            'columns': columns,
            'row_count': row_count
        }
    
    conn.close()
    return db_info

# SQL 쿼리 실행하여 DataFrame 반환
def execute_query_to_df(db_path, query):
    """SQL 쿼리를 실행하여 DataFrame을 반환합니다."""
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"쿼리 실행 오류: {e}")
        return None

# 그래프를 이미지로 변환하는 함수
def fig_to_image_bytes(fig):
    """Plotly 또는 Matplotlib figure를 이미지 bytes로 변환합니다."""
    img_buffer = BytesIO()
    
    # Plotly figure인지 확인
    if PLOTLY_AVAILABLE and hasattr(fig, 'update_layout'):
        try:
            # Plotly figure를 PNG 이미지로 변환
            img_bytes = fig.to_image(format="png", width=1200, height=600)
            img_buffer.write(img_bytes)
            img_buffer.seek(0)
            return img_buffer.getvalue()
        except Exception as e:
            logger.warning(f"Plotly 이미지 변환 실패 (kaleido 필요할 수 있음): {e}")
            # kaleido가 없으면 대체 방법 시도
            try:
                # HTML로 저장 후 변환하는 방법은 복잡하므로, matplotlib로 fallback은 안 됨
                # 대신 사용자에게 kaleido 설치를 안내하거나, matplotlib로 다시 생성
                return None
            except:
                return None
    else:
        # Matplotlib figure
        try:
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            return img_buffer.getvalue()
        except Exception as e:
            logger.error(f"Matplotlib 이미지 변환 오류: {e}")
            return None

# 그래프 생성 함수
def create_chart_from_query(db_path, query, chart_type="auto"):
    """SQL 쿼리 결과를 그래프로 시각화합니다."""
    df = execute_query_to_df(db_path, query)
    if df is None or df.empty:
        return None
    
    # 한글 폰트 설정
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    except:
        try:
            plt.rcParams['font.family'] = 'AppleGothic'  # Mac
        except:
            try:
                plt.rcParams['font.family'] = 'NanumGothic'  # Linux
            except:
                plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 자동으로 차트 타입 결정
    if chart_type == "auto":
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(df.columns) == 2:
            if len(num_cols) == 2:
                # 두 숫자 컬럼: 산점도
                chart_type = "scatter"
            elif len(cat_cols) == 1 and len(num_cols) == 1:
                # 범주형 + 숫자: 막대 그래프
                chart_type = "bar"
        elif len(num_cols) == 1 and len(cat_cols) >= 1:
            chart_type = "bar"
        elif len(num_cols) >= 2:
            chart_type = "line"
        else:
            chart_type = "bar"
    
    # Plotly가 있으면 Plotly 사용, 없으면 matplotlib 사용
    if PLOTLY_AVAILABLE:
        try:
            if chart_type == "bar":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col}별 {y_col}")
                else:
                    return None
            elif chart_type == "line":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_cols = df.columns[1:]
                    fig = go.Figure()
                    for col in y_cols:
                        fig.add_trace(go.Scatter(x=df[x_col], y=df[col], mode='lines+markers', name=col))
                    fig.update_layout(title=f"{x_col}별 추이", xaxis_title=x_col, yaxis_title="값")
                else:
                    return None
            elif chart_type == "pie":
                if len(df.columns) >= 2:
                    labels_col = df.columns[0]
                    values_col = df.columns[1]
                    fig = px.pie(df, names=labels_col, values=values_col, title=f"{labels_col}별 분포")
                else:
                    return None
            elif chart_type == "scatter":
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                else:
                    return None
            else:
                # 기본: 막대 그래프
                if len(df.columns) >= 2:
                    x_col = df.columns[0]
                    y_col = df.columns[1]
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{x_col}별 {y_col}")
                else:
                    return None
            
            fig.update_layout(height=500)
            return fig
        except Exception as e:
            logger.error(f"Plotly 그래프 생성 오류: {e}")
            # Plotly 실패 시 matplotlib로 fallback
    
    # matplotlib로 그래프 생성
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "bar":
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1]
                ax.bar(df[x_col].astype(str), df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col}별 {y_col}")
                plt.xticks(rotation=45, ha='right')
        elif chart_type == "line":
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_cols = df.columns[1:]
                for col in y_cols:
                    ax.plot(df[x_col], df[col], marker='o', label=col)
                ax.set_xlabel(x_col)
                ax.set_ylabel("값")
                ax.set_title(f"{x_col}별 추이")
                ax.legend()
                plt.xticks(rotation=45, ha='right')
        elif chart_type == "pie":
            if len(df.columns) >= 2:
                labels_col = df.columns[0]
                values_col = df.columns[1]
                ax.pie(df[values_col], labels=df[labels_col].astype(str), autopct='%1.1f%%')
                ax.set_title(f"{labels_col}별 분포")
        elif chart_type == "scatter":
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1]
                ax.scatter(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col}")
        else:
            # 기본: 막대 그래프
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = df.columns[1]
                ax.bar(df[x_col].astype(str), df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col}별 {y_col}")
                plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"그래프 생성 오류: {e}")
        return None

# 페이지 설정
st.set_page_config(
    page_title="데이터분석 챗봇",
    page_icon="🗄️",
    layout="wide"
)

# 초기 상태 설정
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "clear_chat" not in st.session_state:
    st.session_state.clear_chat = False

# 대화 초기화 플래그 확인
if st.session_state.clear_chat:
    st.session_state.chat_history = []
    st.session_state.clear_chat = False

if "sql_db" not in st.session_state:
    st.session_state.sql_db = None

if "db_path" not in st.session_state:
    st.session_state.db_path = None

if "db_name" not in st.session_state:
    st.session_state.db_name = None

if "db_info" not in st.session_state:
    st.session_state.db_info = None

if "table_names" not in st.session_state:
    st.session_state.table_names = []

if "graph_fig" not in st.session_state:
    st.session_state.graph_fig = None

if "llm_model" not in st.session_state:
    st.session_state.llm_model = "gpt-5.1"

if "db_connections" not in st.session_state:
    st.session_state.db_connections = []

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
.stButton > button {
    background-color: #ff69b4 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 0.5rem 1rem !important;
    font-weight: bold !important;
}

.stButton > button:hover {
    background-color: #ff1493 !important;
}
</style>
""", unsafe_allow_html=True)

# 제목 영역
st.markdown("""
<div style="text-align: center; margin-top: -3rem; margin-bottom: 1rem;">
    <h1 style="font-size: 7rem; font-weight: bold; margin: 0; line-height: 1.2;">
        <span style="color: #1f77b4;">데이터분석</span> 
        <span style="color: #ffd700;">챗봇</span>
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("모델을 선택하고 CSV 또는 엑셀 파일을 업로드해주세요.")

# 사이드바 설정
with st.sidebar:
    # 1. LLM 모델 선택
    st.markdown('<h2 style="color: #1f77b4;">1. LLM 모델 선택</h2>', unsafe_allow_html=True)
    all_models = ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"]
    
    if 'llm_model' not in st.session_state:
        st.session_state.llm_model = all_models[0]
    
    try:
        current_index = all_models.index(st.session_state.llm_model)
    except ValueError:
        current_index = 0
    
    selected_model = st.radio(
        "사용할 언어모델을 선택하세요",
        options=all_models,
        index=current_index,
        key='llm_model_radio'
    )
    st.session_state.llm_model = selected_model

    # 2. CSV/엑셀 파일 업로드
    st.markdown('<h2 style="color: #ffd700;">2. 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "CSV 또는 엑셀 파일을 선택하세요 (여러 파일 선택 가능)", 
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # 업로드된 파일 목록 표시
        st.markdown("**업로드된 파일:**")
        for file in uploaded_files:
            file_size = len(file.getvalue()) / 1024  # KB
            st.text(f"  - {file.name} ({file_size:.1f} KB)")
        
        process_button = st.button("데이터베이스 생성하기")
        
        if process_button:
            with st.spinner(f"{len(uploaded_files)}개 파일을 데이터베이스로 변환 중입니다..."):
                try:
                    # 모든 열려있는 DB 닫기
                    close_all_databases()
                    
                    # DB 이름 생성 (첫 번째 파일명 사용 또는 통합 DB)
                    if len(uploaded_files) == 1:
                        db_name = os.path.splitext(uploaded_files[0].name)[0] + ".db"
                    else:
                        db_name = "통합데이터베이스.db"
                    # 특수문자 처리
                    db_name = re.sub(r'[^a-zA-Z0-9_.]', '_', db_name)
                    
                    # 여러 파일을 DB로 변환
                    db_path, table_names = files_to_db(uploaded_files, db_name)
                    
                    # DB 정보 가져오기
                    db_info = get_db_info(db_path)
                    
                    # SQLDatabase 객체 생성
                    db_uri = f"sqlite:///{db_path}"
                    sql_db = SQLDatabase.from_uri(db_uri)
                    
                    # 세션 상태 업데이트
                    st.session_state.db_path = db_path
                    st.session_state.db_name = db_name
                    st.session_state.db_info = db_info
                    st.session_state.table_names = table_names
                    st.session_state.sql_db = sql_db
                    
                    st.success(f"데이터베이스 '{db_name}' 생성 완료! ({len(table_names)}개 테이블)")
                    logger.info(f"데이터베이스 생성 완료: {db_name}, 테이블 수: {len(table_names)}")
                    
                except Exception as e:
                    st.error(f"데이터베이스 생성 중 오류가 발생했습니다: {str(e)}")
                    logger.error(f"데이터베이스 생성 오류: {e}")

    # 연결된 데이터베이스 정보 표시
    if st.session_state.db_name:
        st.markdown('<h3 style="color: #ff69b4;">연결된 데이터베이스</h3>', unsafe_allow_html=True)
        st.text(f"이름: {st.session_state.db_name}")
        
        if st.session_state.db_info:
            st.markdown('<h4 style="color: #1f77b4;">데이터베이스 정보</h4>', unsafe_allow_html=True)
            for table, info in st.session_state.db_info.items():
                st.markdown(f"**테이블: {table}**")
                st.text(f"  - 컬럼 수: {len(info['columns'])}")
                st.text(f"  - 컬럼명: {', '.join(info['columns'])}")
                st.text(f"  - 전체 행 수: {info['row_count']:,}")
                st.markdown("---")

    # 대화 초기화 버튼
    clear_button = st.button("대화 초기화", key="clear_chat_button")
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.clear_chat = True
        st.rerun()
    
    # 현재 설정 표시
    st.markdown('<h3 style="color: #1f77b4;">현재 설정</h3>', unsafe_allow_html=True)
    st.text(f"모델: {st.session_state.llm_model}")
    if st.session_state.db_name:
        st.text(f"데이터베이스: {st.session_state.db_name}")
    st.text(f"대화 기록: {len(st.session_state.chat_history)}개")

# 메인 화면에 데이터베이스 정보 표시
if st.session_state.db_info:
    st.markdown("### 📊 데이터베이스 정보")
    for table, info in st.session_state.db_info.items():
        with st.expander(f"테이블: **{table}**"):
            st.markdown(f"**컬럼 수:** {len(info['columns'])}")
            st.markdown(f"**컬럼명:** {', '.join(info['columns'])}")
            st.markdown(f"**전체 행 수:** {info['row_count']:,}")

# 대화 내용 표시
for i, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.markdown(message["content"])
        else:
            st.write(message["content"])
        
        # 그래프가 있으면 표시
        if message["role"] == "assistant" and "graph_fig" in st.session_state and i == len(st.session_state.chat_history) - 1:
            if st.session_state.graph_fig is not None:
                # Plotly인지 matplotlib인지 확인
                if PLOTLY_AVAILABLE and hasattr(st.session_state.graph_fig, 'update_layout'):
                    st.plotly_chart(st.session_state.graph_fig, use_container_width=True)
                else:
                    # matplotlib figure
                    st.pyplot(st.session_state.graph_fig)
                
                # 그래프 다운로드 버튼 추가
                try:
                    img_bytes = fig_to_image_bytes(st.session_state.graph_fig)
                    if img_bytes:
                        # 파일명 생성 (현재 시간 기반)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        filename = f"graph_{timestamp}.png"
                        
                        st.download_button(
                            label="📥 그래프 이미지 다운로드",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/png",
                            key=f"download_graph_{i}"
                        )
                except Exception as e:
                    logger.warning(f"그래프 다운로드 버튼 생성 실패: {e}")
                
                st.session_state.graph_fig = None  # 한 번만 표시

# 사용자 입력 영역
if prompt := st.chat_input("질문을 입력하세요"):
    # 데이터베이스가 연결되어 있는지 확인
    if st.session_state.sql_db is None:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "먼저 CSV 또는 엑셀 파일을 업로드하고 데이터베이스를 생성해주세요."
        })
        st.rerun()
    else:
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # SQL Agent로 답변 생성
        try:
            # LLM 생성
            llm = get_llm(st.session_state.llm_model, temperature=0)
            
            # SQL Agent 생성 및 실행
            with st.spinner("SQL 쿼리를 생성하고 실행 중입니다. 잠시만 기다려주세요..."):
                # 데이터베이스 스키마 정보 가져오기
                db_schema = st.session_state.sql_db.get_table_info()
                
                # 한국어 지원을 강화한 질문 구성
                enhanced_prompt = f"""다음은 데이터베이스 스키마 정보입니다:

{db_schema}

사용자 질문: {prompt}

중요 지침:
- 반드시 SQL 쿼리를 생성하고 실행하여 답변하세요
- "I don't know" 또는 "알 수 없습니다"라고 답변하지 마세요
- 데이터베이스에서 실제로 조회한 결과를 바탕으로 답변하세요
- 모든 답변은 한국어로 제공하세요
- 숫자, 통계, 집계 결과는 명확하게 표시하세요

위 질문에 대해 SQL 쿼리를 생성하고 실행하여 정확한 답변을 제공하세요."""
                
                agent_executor = create_sql_agent(
                    llm=llm,
                    db=st.session_state.sql_db,
                    agent_type="openai-tools",
                    verbose=False
                )
                
                # Agent 실행
                result = agent_executor.invoke({"input": enhanced_prompt})
                response_text = result.get("output", "답변을 생성할 수 없습니다.")
                
                # 그래프 생성 여부 확인 (사용자가 그래프를 요청했는지)
                graph_keywords = ["그래프", "차트", "시각화", "그려", "보여줘", "표시"]
                need_graph = any(keyword in prompt.lower() for keyword in graph_keywords)
                
                # SQL 쿼리 추출 시도 (Agent의 중간 단계에서)
                graph_fig = None
                if need_graph and st.session_state.db_path:
                    try:
                        # 답변에서 숫자나 통계가 포함되어 있으면 그래프 생성 시도
                        # 간단한 집계 쿼리 생성
                        graph_prompt = f"""사용자 질문: {prompt}
                        
                        위 질문에 대한 답변을 그래프로 그릴 수 있도록 적절한 SQL 쿼리를 생성하세요.
                        SELECT 문만 반환하고, 설명은 포함하지 마세요.
                        예: SELECT 컬럼1, SUM(컬럼2) FROM 테이블 GROUP BY 컬럼1 LIMIT 20"""
                        
                        graph_llm = get_llm(st.session_state.llm_model, temperature=0)
                        graph_query_response = graph_llm.invoke(graph_prompt).content
                        
                        # SQL 쿼리 추출 (```sql ... ``` 또는 직접 쿼리)
                        import re
                        sql_match = re.search(r'```sql\s*(.*?)\s*```', graph_query_response, re.DOTALL)
                        if sql_match:
                            graph_query = sql_match.group(1).strip()
                        else:
                            sql_match = re.search(r'SELECT.*?;', graph_query_response, re.DOTALL | re.IGNORECASE)
                            if sql_match:
                                graph_query = sql_match.group(0).strip()
                            else:
                                graph_query = graph_query_response.strip()
                        
                        # 그래프 생성
                        if graph_query:
                            graph_fig = create_chart_from_query(st.session_state.db_path, graph_query)
                    except Exception as e:
                        logger.warning(f"그래프 생성 실패: {e}")
            
            # 구분선 제거
            response_text = remove_separators(response_text)
            
            # 그래프가 있으면 답변에 추가
            if graph_fig is not None:
                st.session_state.graph_fig = graph_fig
                response_text += "\n\n### 📊 시각화"
            
            # 다음 질문 3개 생성 (DB schema 기반)
            try:
                # DB schema 정보 구성
                schema_info = ""
                for table, info in st.session_state.db_info.items():
                    schema_info += f"\n테이블명: {table}\n"
                    schema_info += f"컬럼명: {', '.join(info['columns'])}\n"
                    schema_info += f"행 수: {info['row_count']:,}\n"
                
                next_questions_prompt = f"""
                다음은 데이터베이스 스키마 정보입니다:
                {schema_info}
                
                사용자가 한 질문: {prompt}
                
                생성된 답변:
                {response_text}
                
                위 데이터베이스 스키마 정보를 참고하여, 실제로 답변 가능한 다음 질문 3개를 생성해주세요.
                
                요구사항:
                - 데이터베이스의 테이블과 컬럼을 명확히 참조하여 실제로 SQL로 답변할 수 있는 질문만 생성
                - 답변 내용을 더 깊이 이해하기 위한 후속 질문
                - 답변에서 언급된 내용을 구체화하거나 확장하는 질문
                - 데이터베이스의 다른 컬럼이나 테이블을 활용할 수 있는 관련 질문
                - 각 질문은 완전한 문장으로 작성하되, 간결하고 명확하게 작성
                - 질문은 번호 없이 순서대로 나열하되, 각 질문은 별도의 줄에 작성
                
                형식:
                질문1
                질문2
                질문3
                
                참고: 질문만 작성하고, 설명이나 추가 텍스트는 포함하지 마세요.
                """
                
                next_questions_llm = get_llm(st.session_state.llm_model, temperature=1)
                next_questions_response = next_questions_llm.invoke(next_questions_prompt).content
                next_questions = [q.strip() for q in next_questions_response.strip().split('\n') 
                                if q.strip() and not q.strip().startswith('#')]
                next_questions = next_questions[:3]
                
                if next_questions:
                    response_text += "\n\n"
                    response_text += "### 💡 다음에 물어볼 수 있는 질문들\n\n"
                    for i, question in enumerate(next_questions, 1):
                        response_text += f"{i}. {question}\n\n"
                    
            except Exception as e:
                logger.warning(f"다음 질문 생성 실패: {e}")
            
            # 대화 기록에 추가
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            error_message = f"SQL Agent 실행 중 오류 발생: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            logger.error(f"SQL Agent 실행 오류: {e}")
        
        # 화면 새로고침
        st.rerun()

