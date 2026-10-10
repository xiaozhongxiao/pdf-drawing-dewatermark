# -*- coding: utf-8 -*-
"""
PDF 图纸去水印 + 去图框（批量脚本）

适用：CAD 出图后整页栅格化的 PDF（每页一张整幅位图），水印为“正片叠底”的灰色斜排文字
      （白底上为 RGB 128 灰，线条压在水印之上）；也兼容少量矢量页（水印为文字对象）。

用法：
  python pdf_qushuiyin.py 输入.pdf [-o 输出.pdf] [--start 5] [--end 94]
                          [--no-watermark] [--no-frame] [--keep-front]
  --start/--end  处理的页码范围（从 1 开始计，含两端）；默认全部
  --keep-front   范围之外的页也原样保留在输出里（默认只输出范围内的页）
  --pad          图框涂白时向内多留的像素（默认 3）

依赖：pip install pymupdf opencv-python-headless numpy
"""
import argparse, gc, os, sys, time
import numpy as np, cv2, pymupdf

GRAY = 128
F = GRAY / 255.0

def odd(n):
    n = int(round(n)); return n if n % 2 else n + 1

# ---------------------------------------------------------------- 去水印
def _label_select(mask, pick, H, step=1024, ov=96):
    """分条带做连通域，按 pick(lab, st, a0, a1) 返回的布尔表选出连通块"""
    out = np.zeros_like(mask)
    for y0 in range(0, H, step):
        a0, a1 = max(0, y0 - ov), min(H, y0 + step + ov)
        sub = mask[a0:a1]
        if not sub.any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        if n <= 1:
            continue
        sel = pick(lab, st, a0, a1)
        sel[0] = False
        if sel.any():
            m = sel[lab].astype(np.uint8)
            e = min(step, H - y0)
            out[y0:y0 + e] |= m[y0 - a0:y0 - a0 + e]
        del lab
    return out

def remove_watermark(img, px_per_mm, stroke_mm=0.2, tol=4, reach_mm=1.2):
    """img: HxWx3 uint8（白底图纸），原地修改。返回 (图, 处理像素数)"""
    H = img.shape[0]
    k = max(3, odd(stroke_mm * px_per_mm))
    core = cv2.inRange(img, (GRAY - tol,) * 3, (GRAY + tol,) * 3) // 255     # 128 中性灰
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    thick = cv2.morphologyEx(core, cv2.MORPH_OPEN, ker)        # 粗笔画（CAD 灰细线被滤掉）
    grow = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 2, k + 2)))
    wm = thick | (core & grow)
    del grow
    # 沿“水印灰 + 深色线条”向外延伸 reach_mm，找被墙线隔开的水印碎片
    dark = (cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2]) < 110).astype(np.uint8)
    R = core | dark; del dark
    reach = wm.copy(); k5 = np.ones((5, 5), np.uint8)
    for _ in range(max(1, int(reach_mm * px_per_mm / 2))):
        reach = cv2.dilate(reach, k5) & R
    del R
    rest = core & (1 - wm)
    eroded = cv2.erode(rest, ker)              # 宽度≥k 的碎片才有腐蚀残留
    lim = 6 * k
    def pick(lab, st, a0, a1):
        n = len(st)
        hit = np.zeros(n, bool); hit[np.unique(lab[(reach[a0:a1] > 0) & (lab > 0)])] = True
        fat = np.zeros(n, bool); fat[np.unique(lab[(eroded[a0:a1] > 0) & (lab > 0)])] = True
        w, h, A = st[:, 2], st[:, 3], st[:, 4]
        L = np.maximum(w, h); S = np.minimum(w, h)
        short = (L <= lim) & (S >= 3) & (A >= 0.25 * L * np.maximum(S, 1))   # 短而不细长：碎片/尖端
        return hit & (fat | short)
    wm |= _label_select(rest, pick, H)
    del rest, eroded, reach, core, thick
    if not wm.any():
        return img, 0
    inside = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, ker).astype(bool)   # 含被线条压住的部分
    halo = cv2.dilate(inside.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    del wm
    a = img[halo].astype(np.float32)
    rec = np.minimum(a / F, 255.0)                    # 反算：原值 = 现值 / 0.502
    keep = (~inside[halo]) & (a.max(1) < GRAY - tol)  # 水印外圈的深色像素 = 原图线条边缘，保留
    rec[keep] = a[keep]
    out = img                                          # 原地修改，省内存
    out[halo] = (rec + 0.5).astype(np.uint8)
    return out, int(halo.sum())

# ---------------------------------------------------------------- 图框
def _clusters(idx, gap=3):
    out = []
    for i in idx:
        if out and i - out[-1][1] <= gap: out[-1][1] = i
        else: out.append([i, i])
    return [tuple(int(v) for v in c) for c in out]

def detect_frame(img, long_frac=0.6, tb_min=0.70):
    """找内框(最粗的长线) + 标题栏左边线。返回 dict 或 None"""
    H, W = img.shape[:2]
    dark = (cv2.min(cv2.min(img[:, :, 0], img[:, :, 1]), img[:, :, 2]) < 110).astype(np.uint8)
    hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * long_frac), 1)))
    rows = _clusters(np.where(hl.sum(1) > W * long_frac)[0]); del hl
    vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * long_frac))))
    cols = _clusters(np.where(vl.sum(0) > H * long_frac)[0]); del vl, dark
    th = lambda c: c[1] - c[0]
    # 取最粗的长线；一样粗时取最靠外的（图框在最外侧）
    def pick(cs, outer_is_small):
        if not cs: return None
        return max(cs, key=lambda c: (th(c), -c[0] if outer_is_small else c[0]))
    top = pick([r for r in rows if r[0] < H * 0.35], True)
    bottom = pick([r for r in rows if r[1] > H * 0.65], False)
    left = pick([c for c in cols if c[0] < W * 0.25], True)
    right = pick([c for c in cols if c[1] > W * 0.85], False)
    if not all([top, bottom, left, right]):
        return None
    cand = [c for c in cols if W * tb_min < c[0] < right[0] - 5]
    return dict(left=left, top=top, right=right, bottom=bottom,
                title_left=cand[0] if cand else None, W=W, H=H)

def keep_rect(fr, pad=3):
    x0 = fr['left'][1] + 1 + pad
    y0 = fr['top'][1] + 1 + pad
    x1 = (fr['title_left'] or fr['right'])[0] - pad
    y1 = fr['bottom'][0] - pad
    return x0, y0, x1, y1

def blank_outside(img, rect):
    x0, y0, x1, y1 = rect
    img[:y0] = 255; img[y1:] = 255; img[:, :x0] = 255; img[:, x1:] = 255
    return img

def png_idat(png):
    """取出 PNG 的 IDAT 数据（即 PDF FlateDecode + Predictor 15 可直接用的数据）"""
    import struct
    pos, out = 8, []
    while pos < len(png):
        n, = struct.unpack(">I", png[pos:pos + 4]); t = png[pos + 4:pos + 8]
        if t == b"IDAT": out.append(png[pos + 8:pos + 8 + n])
        pos += 12 + n
    return b"".join(out)

def write_png_to_xref(doc, xref, png, w, h, colors):
    doc.update_stream(xref, png_idat(png), new=False, compress=False)
    doc.xref_set_key(xref, "Filter", "/FlateDecode")
    doc.xref_set_key(xref, "DecodeParms", f"<</Predictor 15/Colors {colors}/BitsPerComponent 8/Columns {w}>>")
    doc.xref_set_key(xref, "ColorSpace", "/DeviceGray" if colors == 1 else "/DeviceRGB")
    doc.xref_set_key(xref, "BitsPerComponent", "8")
    doc.xref_set_key(xref, "Width", str(w)); doc.xref_set_key(xref, "Height", str(h))
    for k in ("SMask", "Mask", "Decode", "Intent"):
        if doc.xref_get_key(xref, k)[0] != "null":
            doc.xref_set_key(xref, k, "null")

# ---------------------------------------------------------------- 页面处理
def process_raster_page(doc, page, do_wm, do_frame, pad, log):
    xref = page.get_images(full=True)[0][0]
    pix = pymupdf.Pixmap(doc, xref)
    if pix.alpha or pix.n != 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix, 0)
    img = np.asarray(pix.samples_mv).reshape(pix.h, pix.w, 3)   # 直接改写 Pixmap 缓冲区
    ppm = min(img.shape[:2]) / 297.0          # 以图幅短边 297mm 估算 像素/毫米
    msg = []
    if do_wm:
        img, n = remove_watermark(img, ppm); msg.append(f"水印像素 {n}")
    if do_frame:
        fr = detect_frame(img)
        if fr:
            img = blank_outside(img, keep_rect(fr, pad)); msg.append("图框已去除")
        else:
            msg.append("未识别到图框(跳过)")
    # 无彩色则存灰度，体积约为 1/3
    s = img[::5, ::5].astype(np.int16)
    gray = int((s.max(2) - s.min(2)).max()) <= 2
    del s
    # 编码为 PNG，取其 IDAT（zlib+PNG 预测器）直接写回原图像对象：只存一份、内存不累积
    if gray:
        ok, buf = cv2.imencode(".png", np.ascontiguousarray(img[:, :, 1]), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        ok, buf = cv2.imencode(".png", img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6])
    w, h = pix.w, pix.h
    del img, pix
    write_png_to_xref(doc, xref, buf.tobytes(), w, h, 1 if gray else 3)
    del buf
    log(("灰度 " if gray else "彩色 ") + "，".join(msg))

def process_vector_page(page, do_wm, do_frame, pad, log, keyword="新疆建筑设计研究院"):
    msg = []
    if do_wm:
        n = 0
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                t = "".join(s["text"] for s in l["spans"])
                if keyword[:4] in t or "股份有限公司" in t or "设计研究" in t:
                    page.add_redact_annot(pymupdf.Rect(l["bbox"]), fill=False); n += 1
        if n:
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                  text=pymupdf.PDF_REDACT_TEXT_REMOVE)
        msg.append(f"删除水印文字 {n} 行")
    if do_frame:
        dpi = 200
        pm = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)[:, :, :3]
        fr = detect_frame(img)
        if fr:
            x0, y0, x1, y1 = keep_rect(fr, pad)
            sc = 72.0 / dpi; R = page.rect
            ox, oy = R.x0, R.y0
            keep = pymupdf.Rect(ox + x0 * sc, oy + y0 * sc, ox + x1 * sc, oy + y1 * sc)
            for r in (pymupdf.Rect(R.x0, R.y0, R.x1, keep.y0), pymupdf.Rect(R.x0, keep.y1, R.x1, R.y1),
                      pymupdf.Rect(R.x0, keep.y0, keep.x0, keep.y1), pymupdf.Rect(keep.x1, keep.y0, R.x1, keep.y1)):
                page.add_redact_annot(r * page.derotation_matrix, fill=(1, 1, 1))
            page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
                                  graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
                                  text=pymupdf.PDF_REDACT_TEXT_REMOVE)
            msg.append("图框已去除")
        else:
            msg.append("未识别到图框(跳过)")
    log("矢量 " + "，".join(msg))

def _rss():
    try:
        return int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0]) // 1024
    except Exception:
        return -1

def main():
    ap = argparse.ArgumentParser(description="PDF 图纸去水印 + 去图框")
    ap.add_argument("pdf"); ap.add_argument("-o", "--out")
    ap.add_argument("--start", type=int, default=1); ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--no-watermark", action="store_true"); ap.add_argument("--no-frame", action="store_true")
    ap.add_argument("--keep-front", action="store_true"); ap.add_argument("--pad", type=int, default=3)
    a = ap.parse_args()
    src = pymupdf.open(a.pdf)
    end = a.end or len(src)
    s0, s1 = a.start - 1, end - 1
    out = a.out or os.path.splitext(a.pdf)[0] + "_去水印去图框.pdf"
    doc = pymupdf.open()
    if a.keep_front:
        doc.insert_pdf(src); first = s0
    else:
        doc.insert_pdf(src, from_page=s0, to_page=s1); first = 0
    src.close()
    t0 = time.time()
    for i in range(first, first + (s1 - s0 + 1)):
        page = doc[i]; t = time.time()
        log = lambda m: print(f"[{i - first + a.start:>3}] {m}  {time.time() - t:.1f}s  内存{_rss()}MB", flush=True)
        try:
            imgs = page.get_images(full=True)
            if len(imgs) == 1 and page.get_image_bbox(imgs[0]).width > page.rect.width * 0.9:
                process_raster_page(doc, page, not a.no_watermark, not a.no_frame, a.pad, log)
            else:
                process_vector_page(page, not a.no_watermark, not a.no_frame, a.pad, log)
        except Exception as e:
            print(f"[{i - first + a.start:>3}] 出错，保留原样：{e}", flush=True)
        pymupdf.TOOLS.store_shrink(100)          # 清空 MuPDF 解码缓存，防止内存累积
        gc.collect()
    doc.set_metadata({})
    print("保存中…", flush=True)
    doc.save(out, garbage=3, deflate=True)
    print(f"完成：{out}  共 {len(doc)} 页，用时 {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
