import io, os, pandas as pd, numpy as np, streamlit as st, plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Insight Engine", layout="wide")
st.title("AI Insight Engine — Streamlit")

def summarize_clusters_with_llm(summary_df, api_key, model=None):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai 패키지가 필요합니다. requirements.txt에 openai 추가 후 재배포하세요.")
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    # 404 대비: Gemma 우선, 실패 시 Llama로 폴백 (둘 다 Nvidia 무료)
    candidates = [model] if model else ["google/gemma-2-9b-it", "meta/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct-v0.3"]
    rows = []
    for _, r in summary_df.iterrows():
        prompt = f"""당신은 청년 의견 분석가입니다. 아래 클러스터를 Issue / Root Cause / Action 3줄로 요약하세요. 각 항목은 1문장, 한글로 간결하게.

클러스터 {int(r['cluster'])}:
- 키워드: {r['keywords']}
- 대표의견: {r['대표의견']}
- 의견 수: {int(r['count'])}개

형식 (반드시 이 형식):
Issue: ...
Root Cause: ...
Action: ..."""
        last_err = None
        text = None
        for m in candidates:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                text = resp.choices[0].message.content.strip()
                break
            except Exception as e:
                last_err = e
                if "404" in str(e) or "not found" in str(e).lower():
                    continue
                raise
        if text is None:
            raise RuntimeError(f"LLM 호출 실패 (시도 모델: {candidates}): {last_err}")
        # 견고한 파싱: "- Issue:" , "클러스터 0 요약:" 등 변형 대응
        import re
        # 앞의 "클러스터 ... 요약:" 제거
        t = re.sub(r"^.*?Issue\s*:", "Issue:", text, flags=re.DOTALL | re.IGNORECASE)
        def _extract(field, next_field=None):
            pat = rf"{field}\s*:\s*(.*?)(?=(?:Root Cause|Action|Issue)\s*:|$)" if next_field is None else rf"{field}\s*:\s*(.*?)(?={next_field}\s*:)"
            m = re.search(pat, t, flags=re.DOTALL | re.IGNORECASE)
            return re.sub(r"^\s*[-•]\s*", "", m.group(1).strip()) if m else ""
        issue = _extract("Issue", "Root Cause")
        cause = _extract("Root Cause", "Action")
        action = _extract("Action")
        # 폴백: 정규식 실패 시 전체 텍스트를 Issue에 넣고 원인/조치는 비워두지 않음
        if not (issue and cause and action):
            # 줄 단위 재시도
            for line in t.splitlines():
                ll = line.strip().lstrip("-• ").strip()
                if ll.lower().startswith("issue:") and not issue: issue = ll[6:].strip()
                elif ll.lower().startswith("root cause:") and not cause: cause = ll[11:].strip()
                elif ll.lower().startswith("action:") and not action: action = ll[7:].strip()
            if not issue: issue = t.strip()[:120]
            if not cause: cause = "-"
            if not action: action = "-"
        rows.append({"주제": int(r["cluster"]), "핵심 이슈": issue, "원인": cause, "조치": action})
    return pd.DataFrame(rows).sort_values("cluster")

# 사이드바: API 키 - 로그인 방식 (인식 후 입력창 숨김)
if "nvidia_key" not in st.session_state:
    st.session_state.nvidia_key = st.secrets.get("NVIDIA_API_KEY", "") if hasattr(st, "secrets") else ""

with st.sidebar:
    st.subheader("AI 요약")
    if st.session_state.nvidia_key:
        st.success("인증되었습니다 ✓")
        if st.button("키 삭제", key="logout_nvidia"):
            st.session_state.nvidia_key = ""
            st.session_state.pop("ai_summary", None)
            st.rerun()
    else:
        st.caption("키를 입력하면 요약 기능이 켜집니다.")
        _input = st.text_input("API Key", type="password", placeholder="nvapi-...", label_visibility="collapsed", key="nvidia_key_input")
        if st.button("인증", use_container_width=True, key="auth_nvidia"):
            if _input.strip():
                st.session_state.nvidia_key = _input.strip()
                st.rerun()
            else:
                st.warning("키를 입력하세요.")
        st.caption("발급: integrate.nvidia.com")

effective_api_key = st.session_state.nvidia_key
if effective_api_key:
    os.environ["NVIDIA_API_KEY"] = effective_api_key

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = load_model()

KOREAN_STOP_WORDS = {"청년","지역","광주","전남","정보","경우","부분","요즘","실제로","개인적으로","생각합니다","좋겠습니다","어렵다","어렵습니다","필요하다","필요합니다","있으면"}

def read_and_clean_csv(raw):
    df=None
    for enc in ["utf-8-sig","utf-8","cp949"]:
        try: df=pd.read_csv(io.BytesIO(raw), encoding=enc); break
        except UnicodeDecodeError: continue
    if df is None: raise ValueError("인코딩 실패")
    if "text" not in df.columns: raise ValueError(f"'text' 필요: {list(df.columns)}")
    df=df.copy(); df["text"]=df["text"].astype("string").str.strip()
    df=df.dropna(subset=["text"]); df=df[df["text"]!=""]
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)

def build_analysis(raw, k):
    df=read_and_clean_csv(raw)
    emb=model.encode(df["text"].tolist(), batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    km=KMeans(n_clusters=k, random_state=42, n_init="auto"); df["cluster"]=km.fit_predict(emb)
    docs=df.groupby("cluster")["text"].apply(lambda x:" ".join(x)).reset_index().sort_values("cluster")
    vec=TfidfVectorizer(token_pattern=r"(?u)\b[가-힣]{2,}\b", ngram_range=(1,2), stop_words=list(KOREAN_STOP_WORDS))
    tfidf=vec.fit_transform(docs["text"]); feat=vec.get_feature_names_out()
    kw=pd.DataFrame([{"cluster":cid, "keywords":", ".join(feat[tfidf[i].toarray().ravel().argsort()[::-1][:6]])} for i,cid in enumerate(docs["cluster"])])
    pca=PCA(n_components=2, random_state=42); xy=pca.fit_transform(emb); df["x"],df["y"]=xy[:,0],xy[:,1]
    fig=px.scatter(
        df, x="x", y="y", color=df["cluster"].astype(str),
        hover_data={"x": False, "y": False, "text": True, "cluster": True},
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="주제 지도",
        labels={"color": "주제"},
    )
    # hover에는 text/cluster만, x/y 수치 숨김 + 마커 진하게/크게
    fig.update_traces(marker=dict(size=9, opacity=0.95, line=dict(width=0.7, color="white")))
    fig.update_layout(legend_title_text="주제", xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    cnt=df["cluster"].value_counts().sort_index().reset_index(); cnt.columns=["cluster","count"]
    centers=km.cluster_centers_/np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    reps=[{"cluster":c,"대표의견":df.iloc[np.where(df["cluster"]==c)[0][cosine_similarity(emb[np.where(df["cluster"]==c)[0]], [centers[c]]).ravel().argmax()]]["text"]} for c in sorted(df["cluster"].unique())]
    summary=cnt.merge(kw, on="cluster").merge(pd.DataFrame(reps), on="cluster")
    return df, emb, km, summary, fig

if "state" not in st.session_state: st.session_state.state=None
up=st.file_uploader("CSV 파일 업로드", type=["csv"])
k=st.slider("주제 수", 3, 10, 7)
if st.button("분석 시작", type="primary"):
    if up is None: st.error("CSV 파일을 업로드하세요.")
    else:
        with st.spinner("분석 중..."):
            df,emb,km,summary,fig=build_analysis(up.getvalue(), k)
            st.session_state.state={"df":df,"emb":emb,"km":km,"summary":summary,"fig":fig}
            st.success(f"분석 완료: {len(df):,}개 / {k}개 주제")
if st.session_state.state:
    st.dataframe(st.session_state.state["summary"], use_container_width=True, hide_index=True)
    st.plotly_chart(st.session_state.state["fig"], use_container_width=True)
    st.download_button("결과 CSV 다운로드", st.session_state.state["summary"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "result.csv", "text/csv")
    st.divider(); st.subheader("AI 주제 요약")
    if not effective_api_key:
        st.info("사이드바에서 인증하면 요약을 생성할 수 있습니다.")
    if st.button("AI 요약 생성", type="secondary", disabled=not bool(effective_api_key)):
        if not effective_api_key:
            st.warning("사이드바에서 먼저 인증하세요.")
        else:
            with st.spinner("AI 요약 생성 중... (클러스터 수만큼 호출)"):
                try:
                    ai_summary = summarize_clusters_with_llm(st.session_state.state["summary"], effective_api_key)
                    st.session_state["ai_summary"] = ai_summary
                    st.success("요약 완료")
                except Exception as e:
                    st.error(f"요약 실패: {e}")
    if "ai_summary" in st.session_state:
        # 테이블 잘림 방지: 글 줄바꿈 허용 + 컬럼 너비 조정
        st.markdown("""<style>
        [data-testid="stDataFrame"] td { white-space: normal !important; word-break: break-word !important; }
        </style>""", unsafe_allow_html=True)
        st.dataframe(st.session_state["ai_summary"], use_container_width=True, hide_index=True, height=420,
            column_config={
                "주제": st.column_config.NumberColumn("주제", width="small"),
                "핵심 이슈": st.column_config.TextColumn("핵심 이슈", width="large"),
                "원인": st.column_config.TextColumn("원인", width="large"),
                "조치": st.column_config.TextColumn("조치", width="large"),
                # 구버전 키 호환
                "cluster": st.column_config.NumberColumn("주제", width="small"),
                "Issue": st.column_config.TextColumn("핵심 이슈", width="large"),
                "Root Cause": st.column_config.TextColumn("원인", width="large"),
                "Action": st.column_config.TextColumn("조치", width="large"),
            })
        st.download_button("AI 요약 CSV 다운로드", st.session_state["ai_summary"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "ai_summary.csv", "text/csv")
    st.divider(); st.subheader("의미 검색")
    with st.form("search_form", clear_on_submit=False):
        q=st.text_input("검색어", placeholder="예: 취업 지원")
        col1, col2 = st.columns(2)
        with col1: topk=st.slider("결과 수", 3, 10, 5, key="topk")
        with col2: threshold = st.slider("유사도 기준", 0.0, 1.0, 0.40, 0.05, key="threshold")
        submitted = st.form_submit_button("검색", use_container_width=True)
    if submitted:
        if not q.strip():
            st.warning("검색어를 입력하세요.")
        else:
            qemb=model.encode([q], normalize_embeddings=True)
            sims=cosine_similarity(qemb, st.session_state.state["emb"]).ravel()
            idx=sims.argsort()[::-1][:topk]
            res=pd.DataFrame([{"rank":r,"score":float(sims[i]),"cluster":int(st.session_state.state["df"].iloc[i]["cluster"]),"text":st.session_state.state["df"].iloc[i]["text"]} for r,i in enumerate(idx,1)])
            filtered = res[res["score"] >= threshold].reset_index(drop=True)
            if filtered.empty:
                st.warning(f"기준 {threshold:.2f} 이상 없음 (최고 {res['score'].max():.4f})")
                st.dataframe(res, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("유사도", format="%.4f"), "text": st.column_config.TextColumn("의견", width="large"), "cluster": st.column_config.NumberColumn("주제", format="%d"), "rank": st.column_config.NumberColumn("순위", format="%d")})
            else:
                if len(filtered) < len(res):
                    st.info(f"{topk}개 중 {len(filtered)}개 표시")
                st.dataframe(filtered, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("유사도", format="%.4f"), "text": st.column_config.TextColumn("의견", width="large"), "cluster": st.column_config.NumberColumn("주제", format="%d"), "rank": st.column_config.NumberColumn("순위", format="%d")})
