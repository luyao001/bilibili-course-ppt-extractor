# -*- coding: utf-8 -*-
"""把同一章节的多讲PDF按讲次顺序合并为单个带书签的课件PDF

用法示例：
  python merge.py "E:\\课件输出"                  # 合并目录下所有 第N章/ 子目录 -> 合并课件/第N章.pdf
  python merge.py "E:\\课件输出\\第1章" -o out    # 只合并指定章节目录
  python merge.py "E:\\课件输出" --flat           # 每个PDF视为一讲，整体合并为一个文件
"""
import argparse, os, re, sys


def sort_key(path):
    """按文件名中的数字段自然排序：1.2 < 1.10 < 2.1；无数字回退到文件名"""
    name = os.path.splitext(os.path.basename(path))[0]
    runs = tuple(map(int, re.findall(r'\d+', name)))
    return (runs, name)


def collect_pdfs(directory):
    return sorted(
        (os.path.join(directory, f) for f in os.listdir(directory)
         if f.lower().endswith('.pdf')),
        key=sort_key)


def merge_pdfs(pdfs, out_path, titles=None):
    """合并PDF列表，每讲一个书签；返回 (页数, 书签数)"""
    import fitz
    doc = fitz.open()
    toc = []
    for i, p in enumerate(pdfs):
        start = doc.page_count + 1
        with fitz.open(p) as src:
            doc.insert_pdf(src)
        toc.append([1, (titles[i] if titles else
                        os.path.splitext(os.path.basename(p))[0]), start])
    if doc.page_count == 0:
        doc.close()
        raise RuntimeError('没有可合并的页面')
    doc.set_toc(toc)
    doc.save(out_path)
    n, m = doc.page_count, len(toc)
    doc.close()
    return n, m


def main():
    ap = argparse.ArgumentParser(description='按章合并分讲PDF为带书签的课件合集')
    ap.add_argument('root', help='课件根目录（含第N章子目录），或直接指定某个章节目录')
    ap.add_argument('-o', '--out', default=None, help='输出目录（默认 <root>/合并课件）')
    ap.add_argument('--flat', action='store_true',
                    help='把root下的所有PDF直接按序合并为一个文件（不按章分）')
    args = ap.parse_args()

    import fitz
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        sys.exit(f'目录不存在: {root}')

    if args.flat:
        out_dir = args.out or root
        os.makedirs(out_dir, exist_ok=True)
        pdfs = collect_pdfs(root)
        if not pdfs:
            sys.exit('目录下没有PDF')
        out = os.path.join(out_dir, os.path.basename(root) + '_合集.pdf')
        n, m = merge_pdfs(pdfs, out)
        print(f'{out}: {m}讲 {n}页')
        return

    chapters = sorted(
        (os.path.join(root, d) for d in os.listdir(root)
         if os.path.isdir(os.path.join(root, d)) and re.fullmatch(r'第\d+章', d)),
        key=sort_key)
    if not chapters and re.fullmatch(r'第\d+章', os.path.basename(root)):
        chapters = [root]           # 直接指定了某个章节目录
    if not chapters:
        sys.exit('未找到 第N章 子目录（或传入的目录名不是 第N章 格式）')

    out_dir = args.out or os.path.join(root if len(chapters) > 1 else os.path.dirname(root),
                                       '合并课件')
    os.makedirs(out_dir, exist_ok=True)
    for chap in chapters:
        pdfs = collect_pdfs(chap)
        if not pdfs:
            print(f'跳过空目录: {chap}')
            continue
        out = os.path.join(out_dir, os.path.basename(chap) + '.pdf')
        n, m = merge_pdfs(pdfs, out)
        print(f'{out}: {m}讲 {n}页')


if __name__ == '__main__':
    main()
