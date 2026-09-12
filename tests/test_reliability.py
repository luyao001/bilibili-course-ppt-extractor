import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import cv2
import fitz
import numpy as np

from bili_ppt import extractor, audit
from extract import parse_pages


class ReliabilityTests(unittest.TestCase):
    def test_page_ranges_reject_invalid_or_reversed_pieces_before_expanding(self):
        for spec in ['', '1-3-5', 'abc', '1,5-3', '0', '1-999999999', '1,']:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                parse_pages(spec, 10)
        self.assertEqual(parse_pages(' 1-3, 2, 5 ', 10), [1, 2, 3, 5])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def frame(self, name, image):
        path = self.root / name
        self.assertTrue(cv2.imwrite(str(path), image))
        return str(path)

    def test_short_slide_and_last_slide_survive(self):
        dark = np.zeros((90, 160), np.uint8)
        light = np.full_like(dark, 255)
        a = self.frame('a.png', dark)
        b = self.frame('b.png', light)
        segments, _, _ = extractor.segment([a, a, b, a, a], min_run_s=0.5)
        self.assertEqual([idx for _, idx in segments], [0, 2, 3])

    def test_corrupt_frame_does_not_shift_timestamps(self):
        with self.assertRaisesRegex(RuntimeError, '分析帧损坏'):
            extractor.segment([str(self.root / 'missing.png')])

    def test_ffmpeg_failure_is_not_cached(self):
        video = self.root / 'bad.mp4'
        video.write_bytes(b'invalid')
        frames = self.root / 'frames'
        with self.assertRaisesRegex(RuntimeError, '抽帧失败'):
            extractor.extract_frames(str(video), str(frames))
        self.assertFalse((frames / 'complete.json').exists())

    def test_pdf_atomic_failure_and_safe_chapter(self):
        pages = self.root / 'pages'
        pages.mkdir()
        self.frame('pages/001.png', np.full((120, 200, 3), 127, np.uint8))
        pdf, count = extractor.build_pdf(1, '1/unsafe', str(pages), str(self.root / 'out'), str(self.root))
        original = Path(pdf).read_bytes()
        self.assertEqual(count, 1)
        self.assertEqual(Path(pdf).parent.name, '第1章')
        with patch('img2pdf.convert', side_effect=RuntimeError('disk failure')):
            with self.assertRaises(RuntimeError):
                extractor.build_pdf(1, '1/unsafe', str(pages), str(self.root / 'out'), str(self.root))
        self.assertEqual(Path(pdf).read_bytes(), original)

    def test_pdf_numeric_page_order_over_999(self):
        pages = self.root / 'pages'
        pages.mkdir()
        for number, color in [(999, 50), (1000, 200)]:
            self.frame(f'pages/{number}.png', np.full((120, 200, 3), color, np.uint8))
        pdf, _ = extractor.build_pdf(1, 'test', str(pages), str(self.root / 'out'), str(self.root))
        with fitz.open(pdf) as doc:
            means = [np.frombuffer(page.get_pixmap().samples, np.uint8).mean() for page in doc]
        self.assertLess(means[0], means[1])

    def test_audit_reports_corrupt_marker_and_failed_rerun(self):
        pages = self.root / 'pages'
        pages.mkdir()
        (pages / '001.done.json').write_text('{broken', encoding='utf-8')
        info = [{'part': 'test', 'duration': 60}]
        report = audit.audit(str(self.root), str(self.root), info)
        self.assertEqual(report['summary']['issues'], 1)
        report = audit.audit(str(self.root), str(self.root), info, failed_pages=[1])
        self.assertEqual(report['summary']['done'], 0)
        self.assertIn('本次处理失败', report['issues'][0])

    def test_conservative_end_to_end_and_rerun_removes_stale_pages(self):
        video_dir = self.root / 'video'
        video_dir.mkdir()
        video = video_dir / '001.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (960, 540))
        self.assertTrue(writer.isOpened())
        # Dark slide, bright diagram without OCR text, same template with changed content.
        slides = []
        for background, text, color in [(25, 'DARK', 240), (245, 'DIAGRAM', 15), (245, 'REPLACED', 15)]:
            slide = np.full((540, 960, 3), background, np.uint8)
            cv2.putText(slide, text, (80, 270), cv2.FONT_HERSHEY_SIMPLEX, 3, (color,) * 3, 8)
            cv2.rectangle(slide, (80, 320), (850, 450), (color,) * 3, 6)
            slides.append(slide)
            for _ in range(20):
                writer.write(slide)
        writer.release()
        pages = self.root / 'pages' / '001'
        pages.mkdir(parents=True)
        self.frame('pages/001/999.png', slides[0])
        ocr = lambda image: (None, None)
        done = extractor.process_page('BVtest', 1, '1.test', str(self.root), str(self.root / 'out'), ocr=ocr)
        self.assertGreaterEqual(done['pages'], 3)
        self.assertFalse((pages / '999.png').exists())
        with fitz.open(done['pdf']) as doc:
            self.assertEqual(doc.page_count, done['pages'])
            self.assertLess(np.frombuffer(doc[0].get_pixmap().samples, np.uint8).mean(), 100)
        # A missing PDF invalidates the completion marker and regenerates the result.
        Path(done['pdf']).unlink()
        regenerated = extractor.process_page('BVtest', 1, '1.test', str(self.root), str(self.root / 'out'), ocr=ocr)
        self.assertTrue(Path(regenerated['pdf']).exists())
        cli = subprocess.run([sys.executable, 'extract.py', '--video', str(video),
                              '--work', str(self.root / 'cli-work'), '--out', str(self.root / 'cli-out')],
                             cwd=Path(__file__).resolve().parents[1], capture_output=True)
        self.assertEqual(cli.returncode, 0, cli.stderr.decode('utf-8', errors='replace'))
        self.assertEqual(len(list((self.root / 'cli-out').rglob('*.pdf'))), 1)
        # A failed force run must not reuse a previous success marker.
        with patch.object(extractor, 'extract_pages', side_effect=RuntimeError('failed rerun')):
            with self.assertRaises(RuntimeError):
                extractor.process_page('BVtest', 1, '1.test', str(self.root), str(self.root / 'out'), ocr=ocr, force=True)
        self.assertFalse((self.root / 'pages' / '001.done.json').exists())
        self.assertTrue(Path(regenerated['pdf']).exists())


if __name__ == '__main__':
    unittest.main()
