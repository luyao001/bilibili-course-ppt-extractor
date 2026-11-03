# -*- coding: utf-8 -*-
"""核心提取算法：视频 -> 抽帧 -> 翻页切分 -> 内容页筛选 -> 组内峰值合并 -> PDF

处理单个分P的流程：
1. ffmpeg 以 0.5s 间隔抽取 160x90 分析小帧
2. 块级翻页切分：将画面划为 8x4 块，静态掩码块中出现大幅变化判定为翻页
   （噪声=全画面微小变化；讲师运动经时间方差掩码排除；抗低码率压缩抖动）
3. 候选页中点帧提取原图，OCR 筛选课件内容页（剔除片头/校徽/讲师独占/深色转场）
4. 组内峰值合并：同页动画/标注序列（内容只增不减、版面相似）只保留信息最全一帧
   - vanish 指标：前帧内容像素在后帧中消失的比例（排除模板区域与讲师列带）
5. img2pdf 按时间序合成 PDF，归档到 章节/ 文件夹
"""
import cv2, numpy as np, os, sys, json, glob, re, subprocess, shutil, time, difflib, tempfile

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


def make_ocr():
    """创建多进程友好的 RapidOCR：
    - det 输入限宽 960/max：默认 limit_type=min 会把小图强制放大到短边736，
      配合管道内 1.2~2 倍预放大，det 输入可达 2000~4600 像素宽，单次调用慢 50 倍以上；
    - 每会话限 2 线程：多进程并行时避免线程踩踏（实测该机器 2 进程聚合吞吐最高）。
    """
    import rapidocr_onnxruntime.utils as _ru
    from onnxruntime import SessionOptions as _SO
    if getattr(_ru.SessionOptions, '__name__', '') != '_limited_session_options':
        def _limited_session_options():
            opts = _SO()
            opts.intra_op_num_threads = 2
            return opts
        _ru.SessionOptions = _limited_session_options
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(det_model_path='', det_limit_side_len=960, det_limit_type='max')

SAMPLE_FPS = 2.0    # 分析帧采样率（每秒2帧）
MIN_RUN_S = 1.5     # 页面至少稳定显示的秒数（过滤翻页动画过渡）
MERGE_THR = 1.2     # 相邻段合并阈值（64x36 灰度均值差）
VANISH_THR = 0.12   # 组内峰值合并阈值：前帧内容消失比例低于此值视为同页递进
LAYOUT_SIM = 0.8    # 版面相似度阈值
DARK_THR = 200      # 内容像素判定灰度阈值


def log(work_dir, pnum, msg):
    line = f'[P{pnum:03d}] {msg}'
    print(line, flush=True)
    os.makedirs(os.path.join(work_dir, 'logs'), exist_ok=True)
    with open(os.path.join(work_dir, 'logs', 'process.log'), 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def get_duration(video):
    cap = cv2.VideoCapture(video)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return (n / fps) if np.isfinite(n) and np.isfinite(fps) and fps > 0 and n > 0 else 0.0


def extract_frames(video, fdir, sample_fps=SAMPLE_FPS):
    """ffmpeg 抽取分析用小帧；已有帧缓存则复用"""
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError('sample_fps must be positive')
    stat = os.stat(video)
    signature = {'source': os.path.abspath(video), 'size': stat.st_size,
                 'mtime_ns': stat.st_mtime_ns, 'fps': sample_fps, 'version': 2}
    marker = os.path.join(fdir, 'complete.json')
    old = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
    try:
        with open(marker, encoding='utf-8') as f:
            cached = json.load(f)
        if cached['signature'] == signature and cached['count'] == len(old) and old:
            return old
    except (OSError, ValueError, KeyError, TypeError):
        pass
    os.makedirs(fdir, exist_ok=True)
    # Only remove files owned by this frame cache.
    for path in old + ([marker] if os.path.exists(marker) else []):
        os.remove(path)
    cmd = [FFMPEG, '-y', '-v', 'error', '-i', video,
           '-vf', f'fps={sample_fps},scale=160:90', '-q:v', '3',
           os.path.join(fdir, '%06d.jpg')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'抽帧失败: {result.stderr[-2000:]}')
    frames = sorted(glob.glob(os.path.join(fdir, '*.jpg')))
    if not frames:
        raise RuntimeError('视频未产生分析帧')
    with open(marker, 'w', encoding='utf-8') as f:
        json.dump({'signature': signature, 'count': len(frames)}, f)
    return frames


def segment(frames, thr=1.6, sample_fps=SAMPLE_FPS, min_run_s=0.5, conservative=True):
    """块级翻页切分。
    - 时间方差掩码剔除持续变化区域（讲师人像/全局噪声过高时自动退化为不掩码）
    - 切点条件：静态掩码块中最大的两块均值差均>8，或单块>15
      （翻页=PPT区域整块大幅变化；压缩噪声=全画面微小变化，天然区分）
    返回 ([(t, frame_idx)], static_mask, template_mask)
    """
    ims = []
    for f in frames:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f'分析帧损坏: {f}')
        ims.append(cv2.resize(im, (64, 36)).astype(np.float32))
    ims = [x for x in ims if x is not None]
    n = len(ims)
    if not frames:
        raise ValueError('没有分析帧')
    if not np.isfinite(sample_fps) or not np.isfinite(min_run_s) or sample_fps <= 0 or min_run_s < 0:
        raise ValueError('无效采样参数')
    dt = 1.0 / sample_fps
    if n >= 10:
        stack = np.stack(ims)
        std_map = stack.std(axis=0)
        thr_std = max(8.0, float(np.percentile(std_map, 40)) * 0.6)
        static_mask = (std_map < thr_std).astype(np.float32)
        if static_mask.mean() < 0.15:
            static_mask = np.ones_like(static_mask)
        dark_freq = (stack < DARK_THR).mean(axis=0)
        template64 = (dark_freq > 0.85).astype(np.float32)
        if template64.mean() > 0.5:        # 模板区异常多则不排除
            template64 = np.zeros_like(template64)
    else:
        static_mask = np.ones((36, 64), np.float32)
        template64 = np.zeros((36, 64), np.float32)

    if conservative:
        static_mask = np.ones_like(static_mask)
    sm_blk = static_mask.reshape(4, 9, 8, 8)
    blk_static = (sm_blk.mean(axis=(1, 3)) > 0.6)

    def is_cut(a, b):
        bd = np.abs(a - b).reshape(4, 9, 8, 8).mean(axis=(1, 3))
        cand = bd[blk_static] if blk_static.any() else bd.reshape(-1)
        top2 = np.sort(cand.flat)[-2:]
        return bool((top2 > 8).all() or cand.max() > 15)

    segs, start = [], 0
    for k in range(n - 1):
        if is_cut(ims[k], ims[k + 1]):
            segs.append([start, k]); start = k + 1
    segs.append([start, n - 1])
    segs = [[s, e] for s, e in segs if (e - s + 1) * dt >= min_run_s]

    mids = []
    for s, e in segs:
        mi = (s + e) // 2
        im = ims[mi]
        g = np.zeros((72, 128), np.float32) if im is None else cv2.resize(im, (128, 72))
        mids.append((mi, mi * dt, g))
    if not mids:
        return [], static_mask, template64
    merged = [mids[0]]
    for m in mids[1:]:
        if not conservative and float(np.mean(np.abs(merged[-1][2] - m[2]))) < MERGE_THR:
            continue
        merged.append(m)
    return [(t, mi) for mi, t, _ in merged], static_mask, template64


def ocr_text(ocr, img, x0, x1, y0, y1, scale=2.0):
    h, w = img.shape[:2]
    crop = img[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
    if crop.size == 0:
        return ''
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    res, _ = ocr(big)
    return ''.join(r[1] for r in res).replace(' ', '') if res else ''


def edge_density(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Canny(g, 60, 160).mean())


_FACE = None
def get_face():
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _FACE


def extract_pages(video, work_dir, pnum=1, ocr=None, pages_dir=None,
                  mode='conservative', sample_fps=SAMPLE_FPS, min_run_s=0.5):
    """主流程：返回 metas（页面元信息列表）并把页面PNG写入 pages_dir"""
    cv2.setNumThreads(4)
    if ocr is None:
        if mode == 'compact':
            ocr = make_ocr()
        else:
            ocr = lambda image: (None, None)
    pages_dir = pages_dir or os.path.join(work_dir, 'pages', f'{pnum:03d}')
    os.makedirs(pages_dir, exist_ok=True)

    dur = get_duration(video)
    frames = extract_frames(video, os.path.join(work_dir, 'frames', f'{pnum:03d}'), sample_fps=sample_fps)
    log(work_dir, pnum, f'时长{dur:.0f}s 分析帧{len(frames)}')
    segs, static_mask, template64 = segment(frames, sample_fps=sample_fps, min_run_s=min_run_s, conservative=mode == 'conservative')
    log(work_dir, pnum, f'分段数 {len(segs)} 静态区{static_mask.mean():.0%} 模板区{template64.mean():.0%}')

    with tempfile.TemporaryDirectory(prefix='.candidates-', dir=pages_dir) as candidate_dir:
        cap = cv2.VideoCapture(video)
        cands = []
        for t, fi in segs:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f'原始视频帧读取失败: {t:.2f}s')
            h, w = img.shape[:2]
            topband = cv2.cvtColor(img[int(h*0.02):int(h*0.10), int(w*0.05):int(w*0.95)],
                                   cv2.COLOR_BGR2GRAY)
            top_white = (float(topband.mean()) > 250 and float(topband.std()) < 8)
            title = ocr_text(ocr, img, 0.05, 0.95, 0.03, 0.32)
            mid = ocr_text(ocr, img, 0.0, 1.0, 0.15, 0.75, scale=1.2)
            full = mid if len(mid) >= 10 else ocr_text(ocr, img, 0.0, 1.0, 0.0, 1.0, scale=1.2)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            emap = cv2.Canny(g, 60, 160) > 0
            faces = get_face().detectMultiScale(g, scaleFactor=1.1, minNeighbors=5, minSize=(36, 36))
            fx = faces[:, [0, 2]] * (128.0 / img.shape[1]) if len(faces) else None
            candidate_path = os.path.join(candidate_dir, f'{fi:06d}.png')
            if not cv2.imwrite(candidate_path, img):
                cap.release()
                raise RuntimeError('候选图片写入失败')
            cands.append({'t': t, 'image_path': candidate_path, 'title': title, 'mid_len': len(mid),
                          'full_len': len(full), 'top_white': top_white,
                          'edge': float(emap.mean()), 'edge_map': emap,
                          'darkish': float((g < DARK_THR).mean()),
                          'face_x': (int(fx[:, 0].min()), int((fx[:, 0] + fx[:, 1]).max()))
                                    if fx is not None else None,
                          'gray': cv2.resize(g, (128, 72)).astype(np.float32)})
        cap.release()

        # 内容页筛选：剔除片头白屏页、深色转场页、无正文文字的讲师独占页
        content = [c for c in cands
                   if not c['top_white'] and c['darkish'] < 0.35
                   and (c['mid_len'] >= 10 or c['full_len'] >= 25)]
        if mode == 'conservative':
            # Brightness and OCR failures are not evidence that a slide is disposable.
            content = cands
        if not content:
            content = cands
            log(work_dir, pnum, '警告: 内容页判据全失败，保留全部候选')

        # 讲师列带排除：人脸框中位数中心定位讲师常驻列，不参与内容消失判定
        sm = cv2.resize(static_mask, (128, 72), interpolation=cv2.INTER_NEAREST).astype(bool)
        template128 = cv2.resize(template64, (128, 72), interpolation=cv2.INTER_NEAREST).astype(bool)
        sm &= ~template128
        centers = [(box[0] + box[1]) / 2 for box in
                   (c['face_x'] for c in content if c.get('face_x'))]
        teacher_band = None
        if len(centers) >= 3:
            import statistics
            cc = int(statistics.median(centers))
            cx0, cx1 = max(0, cc - 45), min(128, cc + 45)
            if cx1 - cx0 < 128 * 0.6:
                sm[:, cx0:cx1] = False
                teacher_band = (cx0, cx1)
                log(work_dir, pnum, f'讲师列带排除 {cx0}-{cx1}')

        # 内容区边缘密度：在整帧Canny边缘上屏蔽讲师列带，
        # 使"信息最全"的判定不被教师姿态变化干扰
        valid_cols = None
        if cands and teacher_band:
            W = cands[0]['edge_map'].shape[1]
            valid_cols = np.ones(W, bool)
            valid_cols[int(teacher_band[0] * W / 128):int(teacher_band[1] * W / 128)] = False

        def content_edge(c):
            em = c['edge_map']
            return float(em[:, valid_cols].mean()) if valid_cols is not None else float(em.mean())

        def vanish(a, b):
            darkA = (a['gray'] < DARK_THR) & sm
            darkB = (b['gray'] < DARK_THR) & sm
            return float((darkA & ~darkB).sum()) / max(int(darkA.sum()), 1)

        def layout_sim(a, b):
            ba = (a['gray'] < DARK_THR).reshape(4, 18, 8, 16).mean(axis=(1, 3)) > 0.04
            bb = (b['gray'] < DARK_THR).reshape(4, 18, 8, 16).mean(axis=(1, 3)) > 0.04
            return float((ba == bb).mean())

        # 组内峰值保留：vanish低+版面相似 => 同页递进序列
        groups = []
        for c in content:
            if groups:
                last = groups[-1]['items'][-1]
                if mode == 'compact' and sm.mean() > 0.1 and vanish(last, c) < VANISH_THR and layout_sim(last, c) > LAYOUT_SIM:
                    groups[-1]['items'].append(c)
                    continue
            groups.append({'items': [c]})
        # 保留内容区信息最全一帧（同分取更晚的帧）
        kept = []
        for g in groups:
            _bi, _bc = max(enumerate(g['items']), key=lambda ic: (content_edge(ic[1]), ic[0]))
            kept.append(_bc)

        # 跨组最终态归并：翻页切分可能把同一页的逐步展示切碎成多组，
        # 若相邻保留页满足"旧内容保留(vanish低) + 新增少(new_ratio低) + 版面相似"，
        # 判定为同一页的递进，只保留更完整的一帧——避免半成品页充数
        if mode == 'compact':
            final = []
            for c in kept:
                if final:
                    p = final[-1]
                    dp, dc = (p['gray'] < DARK_THR) & sm, (c['gray'] < DARK_THR) & sm
                    vanish_pc = float((dp & ~dc).sum()) / max(int(dp.sum()), 1)
                    new_ratio = float((dc & ~dp).sum()) / max(int(dc.sum()), 1)
                    if vanish_pc < 0.15 and new_ratio < 0.35 and layout_sim(p, c) > 0.75:
                        if (content_edge(c), 1) >= (content_edge(p), 0):
                            final[-1] = c
                        continue
                final.append(c)
            if len(final) < len(kept):
                log(work_dir, pnum, f'跨组归并 {len(kept)} -> {len(final)}')
            kept = final
        log(work_dir, pnum, f'候选{len(cands)} -> 内容页{len(content)} -> 页面{len(kept)}')

        metas, bad = [], 0
        for i, c in enumerate(kept, 1):
            img = cv2.imread(c['image_path'])
            if img is None:
                raise RuntimeError('候选图片读取失败')
            h, w = img.shape[:2]
            gstd = float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std())
            healthy = (w >= 800 and gstd > 9)
            if not healthy:
                bad += 1
            if not cv2.imwrite(os.path.join(pages_dir, f'{i:03d}.png'), img):
                raise RuntimeError('页面图片写入失败')
            metas.append({'idx': i, 't': round(c['t'], 1), 'title': c['title'][:40],
                          'gray_std': round(gstd, 1), 'healthy': bool(healthy)})
        return metas, bad


def build_pdf(pnum, part, pages_dir, out_dir, work_dir, ocr=None):
    """将页面PNG合成PDF并按章节归档，返回 (pdf_path, n_pages)"""
    import img2pdf
    match = re.match(r'^(\d+)', part)
    chap = match.group(1) if match else '未分类'
    chapdir = os.path.join(out_dir, f'第{chap}章')
    os.makedirs(chapdir, exist_ok=True)
    safe = re.sub(r'[\\/:*?"<>| ]+', '_', part).strip('_')
    pdf_path = os.path.join(chapdir, f'P{pnum:02d}_{safe}.pdf')
    imgs = sorted(glob.glob(os.path.join(pages_dir, '*.png')),
                  key=lambda path: (0, int(os.path.splitext(os.path.basename(path))[0]))
                  if os.path.splitext(os.path.basename(path))[0].isdigit() else (1, path))
    if not imgs:
        raise RuntimeError('没有页面，拒绝生成空 PDF')
    fd, temporary = tempfile.mkstemp(suffix='.pdf', dir=chapdir)
    try:
        with os.fdopen(fd, 'wb') as f:
            img2pdf.convert(imgs, outputstream=f)
        import fitz
        with fitz.open(temporary) as doc:
            if doc.page_count != len(imgs):
                raise RuntimeError('PDF 页数校验失败')
        os.replace(temporary, pdf_path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return pdf_path, len(imgs)


def process_page(bvid, page, part, work_dir, out_dir, ocr=None, cookies=None,
                 force=False, mode='conservative', sample_fps=SAMPLE_FPS, min_run_s=0.5,
                 video_path=None):
    """处理单个分P（断点续跑：已有完成标记且未 force 则跳过）"""
    pnum = page
    pages_dir = os.path.join(work_dir, 'pages', f'{pnum:03d}')
    done_mark = os.path.join(work_dir, 'pages', f'{pnum:03d}.done.json')
    video = video_path or os.path.join(work_dir, 'video', f'{pnum:03d}.mp4')
    if mode not in ('conservative', 'compact'):
        raise ValueError('未知提取模式')
    os.makedirs(os.path.dirname(pages_dir), exist_ok=True)
    config = {'bvid': bvid, 'part': part, 'out': os.path.abspath(out_dir),
              'mode': mode, 'sample_fps': sample_fps, 'min_run_s': min_run_s, 'version': 2}
    if os.path.exists(video):
        stat = os.stat(video)
        config['source'] = [stat.st_size, stat.st_mtime_ns]
    if os.path.exists(done_mark) and not force:
        try:
            import fitz
            with open(done_mark, encoding='utf-8') as f:
                done = json.load(f)
            if done.get('config') == config:
                with fitz.open(done['pdf']) as doc:
                    if doc.page_count == done['pages'] and doc.page_count > 0:
                        log(work_dir, pnum, '已完成且校验通过，跳过')
                        return done
        except (OSError, ValueError, KeyError, RuntimeError, TypeError, AttributeError):
            pass
    if not os.path.exists(video):
        log(work_dir, pnum, '视频不存在')
        return None

    lock = pages_dir + '.lock'
    try:
        os.makedirs(lock, exist_ok=False)
    except FileExistsError:
        raise RuntimeError(f'分P正在处理或上次异常退出，请确认进程已停止后移除锁: {lock}')
    try:
        if os.path.exists(done_mark):
            os.replace(done_mark, done_mark + '.previous')
        with tempfile.TemporaryDirectory(prefix=f'{pnum:03d}-', dir=os.path.dirname(pages_dir)) as staging:
            metas, bad = extract_pages(video, work_dir, pnum, ocr=ocr, pages_dir=staging,
                                       mode=mode, sample_fps=sample_fps, min_run_s=min_run_s)
            pdf_path, n = build_pdf(pnum, part, staging, out_dir, work_dir)
            os.makedirs(pages_dir, exist_ok=True)
            for old in glob.glob(os.path.join(pages_dir, '*.png')):
                os.remove(old)
            for fresh in glob.glob(os.path.join(staging, '*.png')):
                shutil.copy2(fresh, pages_dir)
        done = {'page': pnum, 'part': part, 'pdf': pdf_path, 'pages': n,
                'bad_imgs': bad, 'meta': metas, 'config': config}
        with open(done_mark + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(done, f, ensure_ascii=False, indent=1)
        os.replace(done_mark + '.tmp', done_mark)
        log(work_dir, pnum, f'完成 -> {pdf_path} ({n}页)')
        return done
    finally:
        if os.path.isdir(lock):
            os.rmdir(lock)
