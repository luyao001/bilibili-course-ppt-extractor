# -*- coding: utf-8 -*-
"""结果审查：PDF完整性 / 空白页 / 页数密度 / 图片健康"""
import os, json, glob, re


def audit(work_dir, out_dir, pages_info, targets=None, failed_pages=None):
    import fitz
    import numpy as np
    targets = list(range(1, len(pages_info) + 1)) if targets is None else targets
    issues, stats = [], []
    for page in targets:
        info = pages_info[page - 1]
        part = info['part']
        if page in (failed_pages or []):
            issues.append(f'P{page:03d} {part}: 本次处理失败，旧结果不计入成功')
            continue
        done_f = os.path.join(work_dir, 'pages', f'{page:03d}.done.json')
        if not os.path.exists(done_f):
            issues.append(f'P{page:03d} {part}: 无完成标记')
            continue
        try:
            with open(done_f, encoding='utf-8') as f:
                d = json.load(f)
            if not isinstance(d.get('pages'), int) or d['pages'] < 0:
                raise ValueError('无效页数')
        except (OSError, ValueError, AttributeError) as error:
            issues.append(f'P{page:03d} {part}: 完成标记损坏 {error}')
            continue
        dur_min = info['duration'] / 60
        density = d['pages'] / max(dur_min, 0.1)
        stats.append({'page': page, 'part': part, 'pages': d['pages'],
                      'dur_min': round(dur_min, 1), 'density': round(density, 2)})
        if d['pages'] == 0:
            issues.append(f'P{page:03d} {part}: 0页')
        if density < 0.18 and d['pages'] < 6:
            issues.append(f"P{page:03d} {part}: 页数密度低({d['pages']}页/{dur_min:.0f}min)，"
                          f"建议人工核对该讲是否以板书/演示为主")
        if d.get('bad_imgs', 0):
            issues.append(f"P{page:03d} {part}: {d['bad_imgs']}张图片健康异常")
        pdf = d.get('pdf')
        if not pdf or not os.path.exists(pdf):
            issues.append(f'P{page:03d} {part}: PDF缺失')
            continue
        try:
            with fitz.open(pdf) as doc:
                if doc.page_count != d['pages']:
                    issues.append(f"P{page:03d} {part}: PDF页数{doc.page_count}!={d['pages']}")
                blank = sum(
                    1 for i in range(doc.page_count)
                    if np.frombuffer(doc[i].get_pixmap(dpi=30).samples, dtype=np.uint8).std() < 5)
                if blank:
                    issues.append(f"P{page:03d} {part}: {blank}张疑似空白页")
        except Exception as e:
            issues.append(f'P{page:03d} {part}: PDF损坏 {e}')
        pngs = glob.glob(os.path.join(work_dir, 'pages', f'{page:03d}', '*.png'))
        if len(pngs) != d['pages']:
            issues.append(f"P{page:03d} {part}: 页面PNG数{len(pngs)}!={d['pages']}")
    summary = {'done': len(stats), 'total': len(targets),
               'total_pages': sum(s['pages'] for s in stats), 'issues': len(issues)}
    report = {'summary': summary, 'stats': stats, 'issues': issues,
              'scope': '检查文件完整性与明显图像异常；不证明视频中所有内容均已提取。',
              'algorithm': 'conservative 默认保留候选页，compact 可能合并递进内容。'}
    out = os.path.join(work_dir, 'audit_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f'审查: 完成 {summary["done"]}/{summary["total"]}, '
          f'总页数 {summary["total_pages"]}, 问题 {len(issues)} 个')
    for x in issues:
        print(' -', x)
    return report
