import io, os, pandas as pd, numpy as np, streamlit as st, plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Insight Engine", layout="wide")
st.title("AI Insight Engine — Streamlit")

def summarize_clusters_with_llm(summary_df, api_key, model="google/gemma-2-9b-it"):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai 패키지가 필요합니다. requirements.txt에 openai 추가 후 재배포하세요.")
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
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
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()
        issue = cause = action = ""
        for line in text.splitlines():
            if line.startswith("Issue:"): issue = line.replace("Issue:", "").strip()
            elif line.startswith("Root Cause:"): cause = line.replace("Root Cause:", "").strip()
            elif line.startswith("Action:"): action = line.replace("Action:", "").strip()
        if not (issue and cause and action):
            issue = text[:100]
        rows.append({"cluster": int(r["cluster"]), "Issue": issue, "Root Cause": cause, "Action": action})
    return pd.DataFrame(rows).sort_values("cluster")

# 사이드바: API 키 입력 (깃에 노출 방지 - Secrets 또는 입력창)
with st.sidebar:
    st.subheader("AI 요약 설정")
    st.caption("키 입력은 메모리에만 저장되고 깃에 올라가지 않습니다.")
    secret_key = st.secrets.get("NVIDIA_API_KEY", "") if hasattr(st, "secrets") else ""
    user_key = st.text_input("NVIDIA API Key", type="password", placeholder="nvapi-...", help="https://integrate.nvidia.com 에서 발급. 비워두면 Secrets의 키를 사용합니다.", key="nvidia_key_input")
    effective_api_key = (user_key.strip() if user_key else secret_key)
    if effective_api_key:
        st.success("API 키 인식됨")
        os.environ["NVIDIA_API_KEY"] = effective_api_key
    else:
        st.info("키를 입력하면 주제 요약 기능이 활성화됩니다.")
    st.divider()
    st.caption("모델: google/gemma-2-9b-it (무료)")

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
up=st.file_uploader("CSV 파일 업로드", type=["csv"], help="text 컬럼이 있는 CSV를 올려주세요")
k=st.slider("주제 수 (k)", 3, 10, 7, help="클러스터 개수. 7이 기본값")
if st.button("분석 시작", type="primary"):
    if up is None: st.error("CSV 파일을 업로드하세요.")
    else:
        with st.spinner("분석 중..."):
            df,emb,km,summary,fig=build_analysis(up.getvalue(), k)
            st.session_state.state={"df":df,"emb":emb,"km":km,"summary":summary,"fig":fig}
            st.success(f"분석 완료: {len(df):,}개 의견 / {k}개 주제로 분류")
if st.session_state.state:
    st.dataframe(st.session_state.state["summary"], use_container_width=True, hide_index=True)
    st.plotly_chart(st.session_state.state["fig"], use_container_width=True)
    st.download_button("결과 CSV 다운로드", st.session_state.state["summary"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "result.csv", "text/csv")
    st.divider(); st.subheader("AI 주제 요약 (Issue / 원인 / 조치)")
    st.caption("Nvidia Gemma 무료 모델로 클러스터별 핵심을 자동 요약합니다. API 키가 있으면 활성화됩니다.")
    if st.button("AI 요약 생성", type="secondary"):
        if not effective_api_key:
            st.warning("사이드바에 NVIDIA API Key를 입력하거나 Streamlit Secrets에 등록하세요.")
        else:
            with st.spinner("AI 요약 생성 중... (클러스터 수만큼 호출)"):
                try:
                    ai_summary = summarize_clusters_with_llm(st.session_state.state["summary"], effective_api_key)
                    st.session_state["ai_summary"] = ai_summary
                    st.success("요약 완료")
                except Exception as e:
                    st.error(f"요약 실패: {e}")
    if "ai_summary" in st.session_state:
        st.dataframe(st.session_state["ai_summary"], use_container_width=True, hide_index=True)
        st.download_button("AI 요약 CSV 다운로드", st.session_state["ai_summary"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "ai_summary.csv", "text/csv")
    st.divider(); st.subheader("의미 검색")
    st.caption("찾고 싶은 내용을 문장으로 입력하면 비슷한 의견만 골라줍니다. 사투리로 입력해도 됩니다.")
    with st.form("search_form", clear_on_submit=False):
        q=st.text_input("검색어", placeholder="예: 취업 지원 (사투리 예: 취업할라 카는데)")
        col1, col2 = st.columns(2)
        with col1: topk=st.slider("검색 결과 수", 3, 10, 5, key="topk")
        with col2: threshold = st.slider("유사도 기준값 (낮을수록 많이 표시)", 0.0, 1.0, 0.40, 0.05, key="threshold", help="이 점수 미만은 숨김. 사투리는 점수가 낮게 나와 0.30~0.35로 낮추면 잘 나옵니다.")
        submitted = st.form_submit_button("검색", use_container_width=True)
        st.caption("엔터 키로도 검색됩니다.")
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
                st.warning(f"기준값 {threshold:.2f} 이상 결과가 없습니다. 기준값을 낮춰보세요. (가장 높은 점수: {res['score'].max():.4f})")
                st.dataframe(res, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("유사도", format="%.4f"), "text": st.column_config.TextColumn("의견", width="large"), "cluster": st.column_config.NumberColumn("주제", format="%d"), "rank": st.column_config.NumberColumn("순위", format="%d")})
                st.caption("참고: 사투리나 구어체는 점수가 표준어보다 낮게 나옵니다.")
            else:
                if len(filtered) < len(res):
                    st.info(f"{topk}개 중 {len(filtered)}개만 기준값 {threshold:.2f} 이상")
                st.dataframe(filtered, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("유사도", format="%.4f"), "text": st.column_config.TextColumn("의견", width="large"), "cluster": st.column_config.NumberColumn("주제", format="%d"), "rank": st.column_config.NumberColumn("순위", format="%d")})
