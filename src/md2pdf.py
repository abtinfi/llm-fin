"""
Markdown -> PDF با چیدمان راست‌به‌چپ.

  python src/md2pdf.py GOZARESH.md -o GOZARESH.pdf

چرا WeasyPrint و نه یک هدلس‌براوزر: WeasyPrint متن را با Pango می‌چیند، و Pango
هم الگوریتم دوجهتهٔ یونیکد و shaping فارسی را درست انجام می‌دهد. یعنی جمله‌ای که
وسطش «Causal Consistency» یا «0.767» آمده، بدون هیچ دخالت دستی سرِ جای خودش
می‌نشیند. هیچ کتابخانهٔ reshaping لازم نیست و نباید هم استفاده شود -- آن‌ها متن را
از قبل وارونه می‌کنند و بعد Pango دوباره وارونه‌اش می‌کند.

فونت Vazirmatn از assets/fonts خوانده و در خود PDF جاسازی می‌شود، پس فایل روی هر
کامپیوتری یکسان باز می‌شود حتی اگر فونت فارسی نصب نداشته باشد.
"""

import argparse
import base64
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "assets" / "fonts"


def font_face(name, file, weight):
    """فونت را به‌صورت data URI داخل CSS می‌گذارد تا در PDF جاسازی شود."""
    p = FONT_DIR / file
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f"@font-face{{font-family:'{name}';font-weight:{weight};"
            f"font-style:normal;"
            f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}\n")


CSS = """
@page{
  size:A4;
  margin:18mm 16mm 20mm;
  @bottom-center{
    content:counter(page);
    font-family:'Vazirmatn';font-size:8.5pt;color:#7A8C93;
  }
}
html{direction:rtl;}
body{
  direction:rtl;text-align:right;
  font-family:'Vazirmatn','DejaVu Sans',sans-serif;
  font-size:10pt;line-height:1.85;color:#16242B;
}
h1{font-size:19pt;font-weight:700;line-height:1.45;margin:0 0 .5em;
   color:#0E3238;}
h2{font-size:13.5pt;font-weight:700;margin:1.6em 0 .5em;
   padding-bottom:.25em;border-bottom:2px solid #16242B;
   break-after:avoid;page-break-after:avoid;}
h3{font-size:11pt;font-weight:700;margin:1.2em 0 .35em;color:#0E3238;
   break-after:avoid;page-break-after:avoid;}
p{margin:0 0 .7em;}
strong{font-weight:700;}
hr{border:none;border-top:1px solid #C9D3D5;margin:1.4em 0;}

/* اعداد و اصطلاحات لاتین: ترتیبشان را الگوریتم دوجهته تعیین می‌کند،
   فقط قلمشان عوض می‌شود تا رقم‌ها هم‌عرض باشند. */
code{
  font-family:'DejaVu Sans Mono',monospace;font-size:8.6pt;
  background:#EAF3F3;color:#0B5860;padding:.08em .3em;border-radius:2px;
}

ul,ol{margin:0 0 .8em;padding-right:1.3em;padding-left:0;}
li{margin-bottom:.3em;}

blockquote{
  margin:.9em 0;padding:.6em 1em;border-right:3px solid #0E6B72;
  background:#EAF3F3;color:#0E3238;font-weight:500;
}
blockquote p{margin:0;}

table{
  border-collapse:collapse;width:100%;margin:.7em 0 1.1em;
  font-size:8.4pt;break-inside:avoid;page-break-inside:avoid;
}
caption{caption-side:top;text-align:right;font-size:8.5pt;color:#4A5C64;
  padding-bottom:.3em;}
th,td{
  border:1px solid #C9D3D5;padding:.32em .5em;text-align:right;
  vertical-align:top;
}
thead th{
  background:#EAF3F3;font-weight:700;font-size:8pt;color:#0E3238;
  white-space:nowrap;
}
tbody tr:nth-child(even){background:#F6F8F8;}

/* جهت هر سلول از محتوای خودش می‌آید، نه از یک حدسِ سطحِ جدول: سلولی که
   حرف فارسی ندارد چپ‌چین و هم‌عرض می‌شود تا ستون عددی قابل مقایسه باشد،
   و سلولی که جملهٔ فارسی است راست‌چین و با قلم متن می‌ماند. */
td.ltr{
  font-family:'DejaVu Sans Mono',monospace;font-size:7.9pt;
  direction:ltr;text-align:left;white-space:nowrap;
}
th.ltr{direction:ltr;text-align:left;}

em{font-style:normal;font-weight:500;}

/* خطی که هیچ حرف فارسی ندارد: چیدمانش لاتین، ولی هم‌چنان راست‌چین */
.ltr{direction:ltr;text-align:right;}
"""


PERSIAN = lambda t: any("\u0600" <= ch <= "\u06FF" for ch in t)


def _ltr_runs(html):
    """
    هر پاراگراف، آیتم لیست یا سلول جدول که هیچ حرف فارسی ندارد، LTR می‌شود.

    چرا لازم است: خطی مثل «128 items / 64 pairs» داخل یک بلوک راست‌به‌چپ، طبق
    الگوریتم دوجهتهٔ یونیکد، عددِ ابتدایش را جدا از کلمات لاتین حل می‌کند و
    نتیجه «items / 64 pairs 128» می‌شود -- عدد می‌پرد ته خط. وقتی کل خط لاتین
    است، جهتش هم باید لاتین باشد؛ آن‌وقت راست‌چین می‌ماند ولی ترتیبش درست است.

    این کار فقط روی خطوطی انجام می‌شود که حتی یک حرف فارسی ندارند، پس جمله‌های
    مخلوط -- که اکثر متن همین‌اند -- دست‌نخورده باقی می‌مانند و ترتیبشان را
    خودِ الگوریتم دوجهته تعیین می‌کند.
    """
    import re

    def fix(m):
        tag, attrs, inner = m.group(1), m.group(2), m.group(3)
        text = re.sub(r"<[^>]+>", "", inner)
        if text.strip() and not PERSIAN(text):
            return f'<{tag}{attrs} class="ltr">{inner}</{tag}>'
        # نقطه/دونقطه بعد از یک کلمهٔ لاتین، کاراکتر خنثی است و جهتش را از
        # همسایه‌هایش می‌گیرد. یک RLM آن را به جملهٔ فارسی می‌چسباند تا وقتی
        # بعدش متن فارسیِ دیگری روی همان خط می‌آید، وسط جمله نپرد.
        # نکته: در انتهای پاراگراف تفاوتی ایجاد نمی‌کند -- آنجا نقطه در هر حال
        # سمت چپ می‌نشیند، که چیدمان درستِ راست‌به‌چپ است، نه اشکال.
        inner = re.sub(r"([A-Za-z0-9_)\]])([.:؛])(\s*)(</|$)",
                       "\\1\u200F\\2\\3\\4", inner)
        return f"<{tag}{attrs}>{inner}</{tag}>"

    return re.sub(r"<(p|li|td|th)([^>]*)>(.*?)</\1>", fix, html, flags=re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    src = Path(args.source)
    out = Path(args.out or src.with_suffix(".pdf"))

    html_body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists", "attr_list"],
    )

    html_body = _ltr_runs(html_body)


    fonts = (font_face("Vazirmatn", "Vazirmatn-Regular.ttf", 400)
             + font_face("Vazirmatn", "Vazirmatn-Medium.ttf", 500)
             + font_face("Vazirmatn", "Vazirmatn-Bold.ttf", 700))

    html = (f'<!doctype html><html lang="fa" dir="rtl"><head>'
            f'<meta charset="utf-8"><style>{fonts}{CSS}</style></head>'
            f'<body>{html_body}</body></html>')

    from weasyprint import HTML
    HTML(string=html, base_url=str(ROOT)).write_pdf(out)
    kb = out.stat().st_size // 1024
    print(f"wrote {out}  ({kb} KB)")



if __name__ == "__main__":
    main()
