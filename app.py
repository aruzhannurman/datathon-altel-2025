import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
import os

# Get FastAPI URL from environment variable (for Docker) or default to localhost
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def display_statistics(stats):
    """Display beautiful statistics with charts and metrics"""
    st.header("📊 Processing Results")

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Comments", stats["total_comments"])
    with col2:
        st.metric(
            "Generated Answers",
            stats["generated_answers"],
            # delta=f"{stats['answer_rate']}% rate",
        )
    with col3:
        st.metric("Clean Comments", stats["clean_comments"])
    with col4:
        st.metric("Filtered Out", stats["toxic_comments"] + stats["spam_comments"])

    # Detailed breakdown
    st.subheader("📈 Comment Analysis Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        # Comment quality pie chart
        quality_data = {
            "Clean": stats["clean_comments"],
            "Toxic": stats["toxic_comments"],
            "Spam": stats["spam_comments"],
        }

        fig_quality = px.pie(
            values=list(quality_data.values()),
            names=list(quality_data.keys()),
            title="Comment Quality Distribution",
            color_discrete_map={
                "Clean": "#00cc44",
                "Toxic": "#ff4444",
                "Spam": "#ff8800",
            },
        )
        st.plotly_chart(fig_quality, use_container_width=True)

    with col2:
        # Answer generation chart
        answer_data = {
            "Answered": stats["generated_answers"],
            "Not Answered": stats["total_comments"] - stats["generated_answers"],
        }

        fig_answers = px.pie(
            values=list(answer_data.values()),
            names=list(answer_data.keys()),
            title="Answer Generation Rate",
            color_discrete_map={"Answered": "#4CAF50", "Not Answered": "#9E9E9E"},
        )
        st.plotly_chart(fig_answers, use_container_width=True)

    # Distribution charts
    if stats.get("label_distribution") and stats.get("tone_distribution"):
        st.subheader("📋 Comment Classifications")

        col1, col2 = st.columns(2)

        with col1:
            # Label distribution
            if stats["label_distribution"]:
                labels_df = pd.DataFrame(
                    list(stats["label_distribution"].items()),
                    columns=["Label", "Count"],
                )
                fig_labels = px.bar(
                    labels_df,
                    x="Label",
                    y="Count",
                    title="Comment Categories",
                    color="Count",
                    color_continuous_scale="viridis",
                )
                st.plotly_chart(fig_labels, use_container_width=True)

        with col2:
            # Tone distribution
            if stats["tone_distribution"]:
                tones_df = pd.DataFrame(
                    list(stats["tone_distribution"].items()), columns=["Tone", "Count"]
                )
                fig_tones = px.bar(
                    tones_df,
                    x="Tone",
                    y="Count",
                    title="Comment Tones",
                    color="Count",
                    color_continuous_scale="plasma",
                )
                st.plotly_chart(fig_tones, use_container_width=True)

    # Summary table
    st.subheader("📋 Summary Table")
    summary_data = {
        "Metric": [
            "Total Comments",
            "Clean Comments",
            "Toxic Comments",
            "Spam Comments",
            "Relevant Comments",
            "Generated Answers",
            "Answer Rate",
        ],
        "Value": [
            str(stats["total_comments"]),
            str(stats["clean_comments"]),
            str(stats["toxic_comments"]),
            str(stats["spam_comments"]),
            str(stats["relevant_comments"]),
            str(stats["generated_answers"]),
            f"{stats['answer_rate']}%",
        ],
    }

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def display_single_comment_analysis(result_data, comment_text):
    """Display beautiful analysis results for a single comment"""
    st.header("🔍 Comment Analysis Results")

    # Display the original comment
    st.subheader("💬 Original Comment")
    st.info(f'"{comment_text}"')

    # Safety Check - Create visual indicators
    st.subheader("🛡️ Safety Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if result_data["is_toxic"]:
            st.error("🚫 Toxic Content Detected")
        else:
            st.success("✅ Content is Safe")

    with col2:
        if result_data["is_spam"]:
            st.warning("⚠️ Spam Detected")
        else:
            st.success("✅ Not Spam")

    with col3:
        relevance = float(result_data["relevance"])
        if relevance >= 0.7:
            st.success(f"🎯 Highly Relevant ({relevance:.1%})")
        elif relevance >= 0.3:
            st.info(f"📝 Moderately Relevant ({relevance:.1%})")
        else:
            st.warning(f"📄 Low Relevance ({relevance:.1%})")

    # Classification Results
    st.subheader("📊 Classification Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Category", result_data["label"])

    with col2:
        tone_emoji = {
            "positive": "😊",
            "negative": "😞",
            "neutral": "😐",
            "mixed": "🤔",
        }
        tone = result_data["tone"]
        emoji = tone_emoji.get(tone.lower(), "🤷")
        st.metric("Tone", f"{emoji} {tone.title()}")

    # Generated Answer Section
    st.subheader("🤖 AI Generated Response")

    if result_data["generated_answer"] and result_data["generated_answer"].strip():
        # Show answer in a nice container
        st.success("✅ Answer Generated Successfully!")

        # Display the answer in a highlighted box
        st.markdown("### 💡 Suggested Response:")
        st.markdown(
            f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin: 10px 0;
        ">
            <p style="margin: 0; font-size: 16px; line-height: 1.5;">
                {result_data["generated_answer"]}
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Copy button functionality
        st.code(result_data["generated_answer"], language=None)

    else:
        st.warning("⚠️ No answer generated")
        if result_data["is_toxic"]:
            st.info("💡 Answer not generated due to toxic content")
        elif result_data["is_spam"]:
            st.info("💡 Answer not generated due to spam detection")
        elif float(result_data["relevance"]) < 0.3:
            st.info("💡 Answer not generated due to low relevance to Altel/Tele2")

    # Detailed Analysis Table
    st.subheader("📋 Detailed Analysis")

    analysis_data = {
        "Analysis Type": ["Safety", "Content Type", "Relevance", "Category", "Tone"],
        "Result": [
            "Safe" if not result_data["is_toxic"] else "Toxic",
            "Clean" if not result_data["is_spam"] else "Spam",
            f"{float(result_data['relevance']):.1%}",
            result_data["label"],
            result_data["tone"].title(),
        ],
        "Status": [
            "✅" if not result_data["is_toxic"] else "🚫",
            "✅" if not result_data["is_spam"] else "⚠️",
            (
                "🎯"
                if float(result_data["relevance"]) >= 0.7
                else "📝" if float(result_data["relevance"]) >= 0.3 else "📄"
            ),
            "📊",
            tone_emoji.get(result_data["tone"].lower(), "🤷"),
        ],
    }

    analysis_df = pd.DataFrame(analysis_data)
    st.dataframe(analysis_df, use_container_width=True, hide_index=True)


st.title("Instagram Comment Responder")

tab1, tab2 = st.tabs(["Process All Comments", "Process Single Comment"])

with tab1:
    st.header("Process All Comments")
    url = st.text_input("Instagram Post URL")
    if st.button("Process"):
        if url:
            response = requests.post(f"{FASTAPI_URL}/process_post", json={"url": url})
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                st.success("Processing started. Task ID: " + task_id)
                progress_bar = st.progress(0)
                status_text = st.empty()
                while True:
                    resp = requests.get(f"{FASTAPI_URL}/progress/{task_id}")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["status"] == "completed":
                            progress_bar.progress(1.0)
                            status_text.text("Completed! 🎉")

                            # Display statistics if available
                            if "statistics" in data:
                                display_statistics(data["statistics"])

                            # For download, since it's server file, need to get the file
                            # But Streamlit download_button needs the data
                            # So, perhaps fetch the file
                            download_resp = requests.get(
                                f"{FASTAPI_URL}/download/{task_id}"
                            )
                            if download_resp.status_code == 200:
                                st.download_button(
                                    "Download XLSX",
                                    download_resp.content,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    file_name="answers.xlsx",
                                )
                            break
                        elif data["status"] == "error":
                            st.error(data["error"])
                            break
                        else:
                            if data.get("total_steps", 0) > 0:
                                percentage = (
                                    data["progress"] / data["total_steps"]
                                ) * 100
                                progress_bar.progress(
                                    data["progress"] / data["total_steps"]
                                )
                                status_text.text(
                                    f"Processing: {percentage:.1f}% complete ({data['total']} comments)"
                                )
                            elif data["total"] > 0:
                                percentage = (data["progress"] / data["total"]) * 100
                                progress_bar.progress(data["progress"] / data["total"])
                                status_text.text(
                                    f"Processing: {percentage:.1f}% complete ({data['total']} comments)"
                                )
                            else:
                                status_text.text("Initializing...")
                    time.sleep(1)
            else:
                st.error("Failed to start processing")

with tab2:
    st.header("Process Single Comment")
    url2 = st.text_input("Instagram Post URL", key="url2")
    comment = st.text_area(
        "Comment",
        placeholder="Enter the comment you want to analyze and get a response for...",
    )

    if st.button("Analyze Comment & Generate Answer", key="gen"):
        if url2 and comment:
            with st.spinner("Analyzing comment and generating answer... ✨"):
                response = requests.post(
                    f"{FASTAPI_URL}/process_comment",
                    json={"url": url2, "comment": comment},
                )
                if response.status_code == 200:
                    result_data = response.json()
                    st.success("Analysis completed! 🎉")

                    # Display beautiful analysis results
                    display_single_comment_analysis(result_data, comment)

                    # Add balloons for successful completion
                    st.balloons()
                else:
                    st.error("Failed to analyze comment")
        else:
            if not url2:
                st.warning("Please enter an Instagram post URL")
            if not comment:
                st.warning("Please enter a comment to analyze")
