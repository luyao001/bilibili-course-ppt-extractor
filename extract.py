# -*- coding: utf-8 -*-
"""命令行入口：从B站课程视频批量提取PPT页面并生成按章节归档的PDF

用法示例：
  python extract.py BV1qFM86PE6c                       # 全部分P
  python extract.py BV1qFM86PE6c --pages 4             # 仅第4个分P
  python extract.py BV1qFM86PE6c --pages 1-10 --workers 4
  python extract.py BV1qFM86PE6c --out "E:\\课件" --cookies cookies.txt
"""
import argparse, os, sys, time, json, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

from bili_ppt import bili_api, extractor, audit


def parse_pages(spec, total):
    if not spec or spec == 'all':
        return list(range(1, total + 1))
    pages = []
    for piece in spec.split(','):
        if '-' in piece:
            a, b = piece.split('-')
            pages.extend(range(int(a), int(b) + 1))
        else:
            pages.append(int(piece))
    if not pages or any(p < 1 or p > total for p in pages):
        raise ValueError(f'分P必须在 1-{total} 范围内，且区间不能倒序')
    return sorted(set(pages))


def page_worker(bvid, page, part, args, ocr):
    video = args.video or os.path.join(args.work, 'video', f'{page:03d}.mp4')
    if not args.video and (not os.path.exists(video) or extractor.get_duration(video) <= 0):
        ok = bili_api.download_page(bvid, page, video, cookies=args.cookies,
                                    max_height=args.max_height)
        if not ok:
            print(f'[P{page:03d}] 下载失败', flush=True)
            return None
    return extractor.process_page(bvid, page, part, args.work, args.out,
                                  ocr=ocr, cookies=args.cookies, force=args.force, mode=args.mode,
                                  sample_fps=args.sample_fps, min_run_s=args.min_run_s,
                                  video_path=args.video)


def main():
    ap = argparse.ArgumentParser(description='从B站课程视频提取PPT页面生成PDF')
    ap.add_argument('bvid', nargs='?', help='视频BV号，如 BV1qFM86PE6c')
    ap.add_argument('--video', help='处理本地视频，不下载或访问 B 站')
    ap.add_argument('--pages', default='all', help='分P范围：all / 4 / 1-10 / 1,3,5')
    ap.add_argument('--out', default='./课件输出', help='PDF输出目录（默认 ./课件输出）')
    ap.add_argument('--work', default='./bili_ppt_work', help='中间文件目录（默认 ./bili_ppt_work）')
    ap.add_argument('--cookies', default=None, help='cookies.txt 路径（提供可解锁更高清晰度）')
    ap.add_argument('--max-height', type=int, default=480, help='最大分辨率高度（默认480）')
    ap.add_argument('--workers', type=int, default=1, help='并行处理进程数（默认1）')
    ap.add_argument('--force', action='store_true', help='强制重跑已完成的分P')
    ap.add_argument('--mode', choices=['conservative', 'compact'], default='conservative', help='默认保留稳定内容页；compact 合并动画并过滤内容')
    ap.add_argument('--sample-fps', type=float, default=2.0)
    ap.add_argument('--min-run-s', type=float, default=0.5)
    args = ap.parse_args()
    import math, re
    if args.video:
        if args.bvid:
            ap.error('BV 号和 --video 只能指定一个')
        if not os.path.isfile(args.video) or extractor.get_duration(args.video) <= 0:
            ap.error('本地视频不存在或无法读取')
        import hashlib
        args.video = os.path.abspath(args.video)
        args.bvid = 'local_' + hashlib.sha256(args.video.encode('utf-8')).hexdigest()[:16]
    elif not re.fullmatch(r'BV[0-9A-Za-z]+', args.bvid or ''):
        ap.error('请输入合法 BV 号或 --video 本地视频路径')
    if args.workers < 1 or args.max_height < 1 or not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or not math.isfinite(args.min_run_s) or args.min_run_s < 0:
        ap.error('workers / max-height / sample-fps 必须为正，min-run-s 必须非负')
    args.work = os.path.join(args.work, args.bvid)
    args.out = os.path.join(args.out, args.bvid)

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    pages_info = ([{'page': 1, 'part': os.path.splitext(os.path.basename(args.video))[0],
                    'duration': extractor.get_duration(args.video)}]
                  if args.video else bili_api.get_pages(args.bvid))
    total = len(pages_info)
    print(f'视频共 {total} 个分P', flush=True)
    try:
        targets = parse_pages(args.pages, total)
    except ValueError as error:
        ap.error(str(error))
    print(f'待处理 {len(targets)} 个分P: {targets[:10]}{"..." if len(targets) > 10 else ""}', flush=True)

    failed_pages = []
    if args.workers <= 1:
        ocr = None
        if args.mode == 'compact':
            from rapidocr_onnxruntime import RapidOCR
            ocr = RapidOCR()
        for page in targets:
            part = pages_info[page - 1]['part']
            try:
                if page_worker(args.bvid, page, part, args, ocr) is None:
                    failed_pages.append(page)
            except Exception as error:
                failed_pages.append(page)
                extractor.log(args.work, page, f'处理失败: {error}')
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(page_worker, args.bvid, page,
                              pages_info[page - 1]['part'], args, None): page
                    for page in targets}
            for fu in as_completed(futs):
                try:
                    if fu.result() is None:
                        failed_pages.append(futs[fu])
                except Exception as error:
                    failed_pages.append(futs[fu])
                    extractor.log(args.work, futs[fu], f'处理失败: {error}')

    report = audit.audit(args.work, args.out, pages_info, targets, failed_pages=failed_pages)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)
    if report['summary']['issues']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
