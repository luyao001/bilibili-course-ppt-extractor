# -*- coding: utf-8 -*-
"""B站API封装：获取视频分P列表、下载指定分P"""
import subprocess, sys, json, re, urllib.request, time

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
    cmd = [sys.executable, "-m", "yt_dlp", "--no-overwrites",
           "--retries", str(retries), "--socket-timeout", "30",
           "--sleep-requests", "1.2", "--proxy", "",
           "-f", f"bv*[vcodec^=avc1]/bv*/b", "-S", f"res:{max_height}",
           "-o", out_path, "--no-progress",
           f"https://www.bilibili.com/video/{bvid}?p={page}"]
    if cookies:
        cmd += ["--cookies", cookies]
    for attempt in range(retries):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and "ERROR" not in (r.stderr or "")[-400:]:
            return True
        time.sleep(8 + attempt * 10)
    return False


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]+', '_', name).strip('_')
