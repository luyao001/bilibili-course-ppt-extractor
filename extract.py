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
    return sorted({p for p in pages if 1 <= p <= total})


def page_worker(bvid, page, part, args, ocr):
    video = os.path.join(args.work, 'video', f'{page:03d}.mp4')
    if not os.path.exists(video) or os.path.getsize(video) < 300000:
        ok = bili_api.download_page(bvid, page, video, cookies=args.cookies,
                                    max_height=args.max_height)
        if not ok:
            print(f'[P{page:03d}] 下载失败', flush=True)
            return None
    return extractor.process_page(bvid, page, part, args.work, args.out,
                                  ocr=ocr, cookies=args.cookies, force=args.force)


def main():
    ap = argparse.ArgumentParser(description='从B站课程视频提取PPT页面生成PDF')
    ap.add_argument('bvid', help='视频BV号，如 BV1qFM86PE6c')
    ap.add_argument('--pages', default='all', help='分P范围：all / 4 / 1-10 / 1,3,5')
    ap.add_argument('--out', default='./课件输出', help='PDF输出目录（默认 ./课件输出）')
    ap.add_argument('--work', default='./bili_ppt_work', help='中间文件目录（默认 ./bili_ppt_work）')
    ap.add_argument('--cookies', default=None, help='cookies.txt 路径（提供可解锁更高清晰度）')
    ap.add_argument('--max-height', type=int, default=480, help='最大分辨率高度（默认480）')
    ap.add_argument('--workers', type=int, default=1, help='并行处理进程数（默认1）')
    ap.add_argument('--force', action='store_true', help='强制重跑已完成的分P')
    args = ap.parse_args()

    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    pages_info = bili_api.get_pages(args.bvid)
    total = len(pages_info)
    print(f'视频共 {total} 个分P', flush=True)
    targets = parse_pages(args.pages, total)
    print(f'待处理 {len(targets)} 个分P: {targets[:10]}{"..." if len(targets) > 10 else ""}', flush=True)

    if args.workers <= 1:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
        for page in targets:
            part = pages_info[page - 1]['part']
            page_worker(args.bvid, page, part, args, ocr)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(page_worker, args.bvid, page,
                              pages_info[page - 1]['part'], args, None): page
                    for page in targets}
            for fu in as_completed(futs):
                fu.result()

    report = audit.audit(args.work, args.out, pages_info, targets)
    print(json.dumps(report['summary'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
