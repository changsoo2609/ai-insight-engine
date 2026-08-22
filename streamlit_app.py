import io, pandas as pd, numpy as np, streamlit as st, plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Insight Engine", layout="wide")
st.title("AI Insight Engine — Streamlit")

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
        title="Topic Map",
        labels={"color": "Topic"},
    )
    # hover에는 text/cluster만, x/y 수치 숨김 + 마커 진하게/크게
    fig.update_traces(marker=dict(size=9, opacity=0.95, line=dict(width=0.7, color="white")))
    fig.update_layout(legend_title_text="Topic", xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    cnt=df["cluster"].value_counts().sort_index().reset_index(); cnt.columns=["cluster","count"]
    centers=km.cluster_centers_/np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    reps=[{"cluster":c,"대표의견":df.iloc[np.where(df["cluster"]==c)[0][cosine_similarity(emb[np.where(df["cluster"]==c)[0]], [centers[c]]).ravel().argmax()]]["text"]} for c in sorted(df["cluster"].unique())]
    summary=cnt.merge(kw, on="cluster").merge(pd.DataFrame(reps), on="cluster")
    return df, emb, km, summary, fig

if "state" not in st.session_state: st.session_state.state=None
up=st.file_uploader("CSV Upload", type=["csv"])
k=st.slider("Number of Topics (k)", 3, 10, 7)
if st.button("Analyze", type="primary"):
    if up is None: st.error("CSV를 업로드하세요.")
    else:
        with st.spinner("분석 중..."):
            df,emb,km,summary,fig=build_analysis(up.getvalue(), k)
            st.session_state.state={"df":df,"emb":emb,"km":km,"summary":summary,"fig":fig}
            st.success(f"분석 완료: {len(df):,}개 / {k}개 토픽")
if st.session_state.state:
    st.dataframe(st.session_state.state["summary"], use_container_width=True, hide_index=True)
    st.plotly_chart(st.session_state.state["fig"], use_container_width=True)
    st.download_button("결과 CSV 다운로드", st.session_state.state["summary"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "result.csv", "text/csv")
    st.divider(); st.subheader("Semantic Search")
    q=st.text_input("Query", placeholder="예: 취업 지원 (사투리도 가능: 취업할라 카는데)"); topk=st.slider("Top-K", 3, 10, 5, key="topk")
    threshold = st.slider("Similarity Threshold (사투리면 0.30~0.45 권장)", 0.0, 1.0, 0.40, 0.05, key="threshold", help="이 점수 미만은 숨김. 사투리는 표준어보다 점수가 낮게 나와 threshold를 낮춰야 합니다.")
    if st.button("Search"):
        if not q.strip():
            st.warning("검색어를 입력하세요.")
        else:
            qemb=model.encode([q], normalize_embeddings=True)
            sims=cosine_similarity(qemb, st.session_state.state["emb"]).ravel()
            idx=sims.argsort()[::-1][:topk]
            res=pd.DataFrame([{"rank":r,"score":float(sims[i]),"cluster":int(st.session_state.state["df"].iloc[i]["cluster"]),"text":st.session_state.state["df"].iloc[i]["text"]} for r,i in enumerate(idx,1)])
            filtered = res[res["score"] >= threshold].reset_index(drop=True)
            if filtered.empty:
                st.warning(f"임계값 {threshold:.2f} 이상 결과가 없습니다. Threshold를 낮춰보세요. (Top-{topk} 최고점: {res['score'].max():.4f})")
                st.dataframe(res, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("score", format="%.4f"), "text": st.column_config.TextColumn("text", width="large")})
                st.caption("참고: 사투리/구어체는 표준어보다 score가 낮게 나옵니다.")
            else:
                if len(filtered) < len(res):
                    st.info(f"Top-{topk} 중 {len(filtered)}건만 Threshold {threshold:.2f} 이상")
                st.dataframe(filtered, use_container_width=True, hide_index=True, column_config={"score": st.column_config.NumberColumn("score", format="%.4f"), "text": st.column_config.TextColumn("text", width="large")})
