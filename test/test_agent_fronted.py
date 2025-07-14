# operator_writer_ui.py  (完整文件)
import json, os, requests, contextlib
from typing import Dict, Any
import streamlit as st

# ---------- 页面设置 ----------
st.set_page_config(page_title="DataFlow-Agent · 写算子", page_icon="🛠️", layout="centered")
st.title("🛠️ DataFlow-Agent · 写算子（Operator Writer）")

# ---------- 工具 ----------
def stop_running_stream():
    resp = st.session_state.pop("resp_obj", None)
    if resp is not None:
        with contextlib.suppress(Exception):
            resp.close()

# ---------- 基本参数 ----------
api_base: str = st.text_input("后端地址", "http://localhost:8000", help="无需带路径")
col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Language", ["zh", "en"], 0)
with col2:
    model = st.text_input("LLM Model", "deepseek-v3")

session_key = st.text_input("sessionKEY", "dataflow_demo")
target = st.text_area("目标（Target）", "我需要一个算子，能够对用户评论进行情感分析并输出积极/消极标签。", height=100)

st.divider()

# ---------- 写算子参数 ----------
json_file  = st.text_input("pipeline JSON", "/mnt/h_h_public/lh/lz/DataFlow/dataflow/example/ReasoningPipeline/pipeline_math_short.json")
py_path    = st.text_input("算子输出路径", "/mnt/h_h_public/lh/lz/DataFlow/test/operator_sentiment.py")
api_key    = st.text_input("DF_API_KEY", "sk-ClnOAuClTqcZSsc5swPFpb98147MCEkJiQBU1Hu69Vty5Jaj", type="password")
chat_api   = st.text_input("DF_API_URL", "http://123.129.219.111:3000/v1/chat/completions")

col3, col4 = st.columns(2)
with col3:
    execute_operator = st.checkbox("执行算子", False)
with col4:
    use_local_model  = st.checkbox("使用本地模型", False)

local_model = st.text_input("本地模型路径", "/mnt/public/model/huggingface/Qwen2.5-7B-Instruct",
                            disabled=not use_local_model)

timeout = st.number_input("超时 (s)", 60, 7200, 3600, 60)
max_debug = st.number_input("最大 Debug 轮数", 1, 20, 5, 1)

# ---------- 组装 Payload ----------
def build_payload() -> Dict[str, Any]:
    return {
        "language": language,
        "target": target,
        "model": model,
        "sessionKEY": session_key,
        "json_file": json_file,
        "py_path": py_path,
        "api_key": api_key,
        "chat_api_url": chat_api,
        "execute_the_operator": execute_operator,
        "use_local_model": use_local_model,
        "local_model_name_or_path": local_model,
        "timeout": timeout,
        "max_debug_round": max_debug,
    }

# ====================== 普通请求 ======================
if st.button("📨 普通请求"):
    payload = build_payload()
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
    try:
        with st.spinner("等待响应…"):
            r = requests.post(f"{api_base}/chatagent", json=payload, timeout=timeout+30)
        st.write(f"HTTP {r.status_code}")
        if r.ok:
            data = r.json()
            st.success("✅ Done")
            st.json(data, expanded=True)
        else:
            st.error(r.text)
    except Exception as e:
        st.error(f"异常: {e}")

# ====================== 流式请求 ======================
if st.button("🚀 流式请求"):
    stop_running_stream()
    payload = build_payload()
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    with st.spinner("连接后端…"):
        resp = requests.post(f"{api_base}/chatagent/stream",
                             json=payload, stream=True, timeout=None)
    if resp.status_code != 200:
        st.error(f"{resp.status_code} – {resp.text}")
    else:
        st.session_state["resp_obj"] = resp
        placeholder = st.empty()
        prog        = st.progress(0.0)
        dots        = ["", ".", "..", "..."]
        dot_idx     = 0
        done        = False
        finished    = 0
        total_tasks = 0      # 会在收到 start 事件时更新

        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                msg = json.loads(raw.removeprefix("data: ").rstrip("\r"))
                evt = msg.get("event")

                if evt == "start":
                    total_tasks += 1
                    placeholder.markdown(f"🛠 **开始任务 `{msg['task']}`**")
                elif evt == "ping":
                    dot_idx = (dot_idx + 1) % 4
                    placeholder.markdown(f"⏳ 后端处理中{dots[dot_idx]}")
                elif evt == "stream":
                    st.write(msg["line"])
                elif evt == "finish":
                    finished += 1
                    st.success(f"✅ `{msg['task']}` 完成 (⏱ {msg['elapsed']:.2f}s)")
                    st.code(json.dumps(msg["result"], ensure_ascii=False, indent=2),
                            language="json")
                    prog.progress(finished / total_tasks if total_tasks else 0.0)
                elif evt == "done":
                    done = True
                    break
                elif evt == "error":
                    st.error(f"任务失败: {msg.get('detail')}")
                    break

            if done:
                prog.progress(1.0)
                placeholder.success("🎉 全部任务完成")
                st.balloons()
            else:
                placeholder.info("ℹ️ 连接结束")
        except requests.exceptions.ChunkedEncodingError:
            # 服务器正常结束但最后块没发完整，直接当 done 处理
            st.info("后端连接已关闭")
        finally:
            stop_running_stream()