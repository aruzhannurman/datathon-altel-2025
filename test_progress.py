#!/usr/bin/env python3
"""
Simple test to debug progress tracking
"""

import requests
import time
import json


def test_progress():
    # Test URL - you can replace this with a real Instagram URL
    test_url = "https://www.instagram.com/p/test123/"

    print("🔄 Starting test...")

    # Start processing
    response = requests.post(
        "http://localhost:8000/process_post", json={"url": test_url}
    )

    if response.status_code != 200:
        print(f"❌ Failed to start processing: {response.status_code}")
        return

    task_id = response.json()["task_id"]
    print(f"✅ Started task: {task_id}")

    # Monitor progress
    while True:
        resp = requests.get(f"http://localhost:8000/progress/{task_id}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"📊 Progress: {json.dumps(data, indent=2)}")

            if data["status"] == "completed":
                print("🎉 Completed!")
                break
            elif data["status"] == "error":
                print(f"❌ Error: {data.get('error')}")
                break
        else:
            print(f"❌ Failed to get progress: {resp.status_code}")
            break

        time.sleep(2)


if __name__ == "__main__":
    test_progress()
