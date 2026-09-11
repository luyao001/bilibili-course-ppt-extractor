# -*- coding: utf-8 -*-
"""B站API封装：获取视频分P列表、下载指定分P"""
import subprocess, sys, json, re, urllib.request, time, os, tempfile

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")


_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # B站国内直连，绕过系统代理


def get_pages(bvid: str) -> list:
    """返回分P列表 [{page, part, duration}]"""
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://www.bilibili.com/"})
    with _opener.open(req, timeout=30) as r:
        data = json.load(r)
    if data.get("code") != 0:
        raise RuntimeError(f"获取视频信息失败: {data.get('message')}")
    pages = data["data"]["pages"]
    return [{"page": p["page"], "part": p["part"], "duration": p["duration"]} for p in pages]


def download_page(bvid: str, page: int, out_path: str, cookies: str = None,
                  max_height: int = 480, retries: int = 3) -> bool:
    """下载单个分P的视频流（默认未登录最高480P）"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist",
           "--retries", str(retries), "--socket-timeout", "30",
           "--sleep-requests", "1.2", "--proxy", "",
           "-f", f"bv*[vcodec^=avc1]/bv*/b", "-S", f"res:{max_height}",
           "--no-progress",
           f"https://www.bilibili.com/video/{bvid}?p={page}"]
    if cookies:
        cmd += ["--cookies", cookies]
    with tempfile.TemporaryDirectory(prefix='bili-download-', dir=os.path.dirname(os.path.abspath(out_path))) as staging:
        temporary = os.path.join(staging, 'video.mp4')
        for attempt in range(retries):
            r = subprocess.run(cmd + ['-o', temporary], capture_output=True, text=True)
            if r.returncode == 0 and os.path.isfile(temporary):
                import cv2
                cap = cv2.VideoCapture(temporary)
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    os.replace(temporary, out_path)
                    return True
            if attempt + 1 < retries:
                time.sleep(8 + attempt * 10)
    return False


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]+', '_', name).strip('_')
