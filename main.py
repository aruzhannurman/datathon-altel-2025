from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from instagrapi import Client
import pandas as pd
import uuid
from typing import Dict
import time
from models.pipeline import ComprehensivePipeline

BASE_DATA_DIR = "./data/"


class Utils:
    @staticmethod
    def save_comments_to_excel(comments, task_id):
        df = pd.DataFrame(comments)
        file_path = f"{BASE_DATA_DIR}{task_id}_comments.xlsx"
        df.to_excel(file_path, index=False)

        return (df, file_path)

    @staticmethod
    def save_results_df_to_excel(results_df, task_id):
        file_path = f"{BASE_DATA_DIR}{task_id}_results.xlsx"
        results_df.to_excel(file_path, index=False)

        return (results_df, file_path)


app = FastAPI()
cl = Client()
cl.login_by_sessionid(
    "8497330613%3AhFFpdI5niRelF0%3A2%3AAYjXX_DSbQ6zR6LWDHe5e8swUDThGQJpuZcOq9J3Qg"
)
pipeline = ComprehensivePipeline()

# Store progress
progress: Dict[str, dict] = {}


def get_comments(media_pk):
    response = cl.media_comments(media_pk, amount=0)
    comments = []

    for comment in response:
        # Get comment details
        comment_data = {
            "id": comment.pk,
            "text": comment.text,
            "username": comment.user.username,
            "timestamp": comment.created_at_utc.strftime("%Y-%m-%d %H:%M:%S"),
            "likes": comment.like_count,
        }
        comments.append(comment_data)

    return comments


def generate_answer(comment_text: str):
    time.sleep(2)
    return "sample answer to: " + comment_text


async def process_comments(task_id: str, url: str):
    try:
        media_pk = cl.media_pk_from_url(url)
        comments = get_comments(media_pk)

        total = len(comments)
        progress[task_id]["total"] = total

        comments_df, _ = Utils.save_comments_to_excel(comments, task_id)

        result_df = await pipeline.process_comments(comments_df, progress[task_id])
        result_df, results_file = Utils.save_results_df_to_excel(result_df, task_id)

        # Calculate statistics
        total_comments = len(result_df)
        toxic_comments = result_df["is_toxic"].sum()
        spam_comments = result_df["is_spam"].sum()
        relevant_comments = (result_df["relevance_altel_tele2"] > 0).sum()
        generated_answers = (result_df["generated_answer"] != "").sum()

        # Get distributions
        label_distribution = result_df["label"].value_counts().to_dict()
        tone_distribution = result_df["tone"].value_counts().to_dict()

        # Store statistics in progress
        progress[task_id]["statistics"] = {
            "total_comments": int(total_comments),
            "toxic_comments": int(toxic_comments),
            "spam_comments": int(spam_comments),
            "relevant_comments": int(relevant_comments),
            "generated_answers": int(generated_answers),
            "clean_comments": int(total_comments - toxic_comments - spam_comments),
            "answer_rate": (
                round((generated_answers / total_comments * 100), 1)
                if total_comments > 0
                else 0
            ),
            "label_distribution": label_distribution,
            "tone_distribution": tone_distribution,
        }

        progress[task_id]["status"] = "completed"
        progress[task_id]["file"] = results_file
    except Exception as e:
        progress[task_id]["status"] = "error"
        progress[task_id]["error"] = str(e)


class ProcessPostRequest(BaseModel):
    url: str


@app.post("/process_post")
async def process_post(request: ProcessPostRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    progress[task_id] = {
        "status": "processing",
        "progress": 0,
        "total": 0,
        "total_steps": 0,
        "current_step": 0,
        "results": [],
    }
    background_tasks.add_task(process_comments, task_id, request.url)
    return {"task_id": task_id}


@app.get("/progress/{task_id}")
def get_progress(task_id: str):
    if task_id not in progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return progress[task_id]


@app.get("/download/{task_id}")
def download(task_id: str):
    if task_id not in progress or progress[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(
        progress[task_id]["file"],
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename="answers.xlsx",
    )


class ProcessCommentRequest(BaseModel):
    url: str
    comment: str


@app.post("/process_comment")
async def process_comment(request: ProcessCommentRequest):

    comment_data = {
        "id": None,
        "text": request.comment,
        "username": None,
        "timestamp": None,
        "likes": None,
    }

    comments_df = pd.DataFrame([comment_data])

    result_df = await pipeline.process_comments(comments_df, {"progress": 0})

    # Convert to native Python types using pandas to_dict() which handles numpy types
    row_dict = result_df.iloc[0].to_dict()

    return {
        "is_toxic": row_dict["is_toxic"],
        "is_spam": row_dict["is_spam"],
        "relevance": row_dict["relevance_altel_tele2"],
        "generated_answer": row_dict["generated_answer"],
        "label": row_dict["label"],
        "tone": row_dict["tone"],
    }
