"""Boot a real emitted Stack-World app and screenshot it driving the live API."""
import subprocess, time, os, sys, urllib.request
from playwright.sync_api import sync_playwright

app_dir = os.path.abspath("output_apps/task_api")
port = 3261
proc = subprocess.Popen(["node", "server.js"], cwd=app_dir,
                        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                        env={"PATH": os.environ.get("PATH", ""), "PORT": str(port)})
try:
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/", timeout=1).read(); break
        except Exception:
            time.sleep(0.2)
    # seed a few records through the real API so the UI has content
    for t in ["write the research report", "ship the figures", "open a PR"]:
        import json
        r = urllib.request.Request(base + "/tasks", method="POST",
                                   data=json.dumps({"title": t}).encode(),
                                   headers={"content-type": "application/json"})
        urllib.request.urlopen(r, timeout=3).read()
    with sync_playwright() as p:
        b = p.chromium.launch(
            executable_path="/ms-playwright/chromium-1228/chrome-linux64/chrome")
        pg = b.new_page(viewport={"width": 760, "height": 720})
        pg.goto(base + "/")
        pg.wait_for_timeout(600)
        pg.screenshot(path="figures/stack_app.png")
        b.close()
    print("wrote figures/stack_app.png")
finally:
    proc.terminate()
    try: proc.wait(timeout=3)
    except Exception: proc.kill()
