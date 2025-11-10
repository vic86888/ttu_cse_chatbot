import re
import urllib.parse
import scrapy
from parsel import Selector
from readability import Document
from html import unescape
from datetime import datetime, date, timedelta

class TTUCSECrawerTestSpider(scrapy.Spider):
    name = "ttu_cse_news"
    allowed_domains = ["cse.ttu.edu.tw"]

    # 最新消息第 1 頁（行動版列表）
    start_urls = ["https://cse.ttu.edu.tw/p/403-1058-61-1.php?Lang=zh-tw"]

    # 只針對「最新消息」的列表/內文規則
    LIST_RE   = re.compile(r"/p/403-1058-61(?:-\d+)?\.php", re.I)
    DETAIL_RE = re.compile(r"/p/406-1058-\d+(?:,r61)?\.php", re.I)
    PAGE_RE   = re.compile(r"-([0-9]+)\.php", re.I)

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "DOWNLOAD_DELAY": 0.25,
        "AUTOTHROTTLE_ENABLED": True,
        "FEED_EXPORT_ENCODING": "utf-8",
    }

    # -------- helpers --------
    @staticmethod
    def _force_https(url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            parts = parts._replace(scheme="https")
        return urllib.parse.urlunsplit(parts)

    @staticmethod
    def _ensure_lang_zh(url: str) -> str:
        """確保 Lang=zh-tw 保持一致，避免抓到英文頁或重覆"""
        parts = urllib.parse.urlsplit(url)
        q = urllib.parse.parse_qs(parts.query)
        q["Lang"] = ["zh-tw"]
        new_q = urllib.parse.urlencode({k: v[0] for k, v in q.items()})
        return urllib.parse.urlunsplit(parts._replace(query=new_q))

    def _norm_date(self, s: str) -> str | None:
        if not s:
            return None
        s = s.strip()
        m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s)
        if not m:
            return None
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    def _to_date(self, s: str | None) -> date | None:
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 「三個月」這裡用 ~92 天近似，夠用且不需額外相依套件
        self.cutoff: date = date.today() - timedelta(days=92)

    # -------- crawl flow --------
    def parse(self, response):
        """
        列表頁：逐卡片抓連結＋日期；若日期早於 cutoff，當頁之後的卡片與後續分頁都不再爬。
        只在需要時，逐頁前進（而不是一次排入全部 80 頁）。
        """
        stop_pagination = False
        any_kept = False

        # 1) 逐卡片：抓 <a> 與 <i class="mdate">（較寬鬆選擇器）
        for card in response.css(".row.listBS, .listBS"):
            href = card.css("a::attr(href)").get() or card.xpath(".//a/@href").get()
            date_txt = card.css(".mdate::text").get() or card.xpath(".//*[contains(@class,'mdate')]/text()").get()
            if not href:
                continue

            list_date_str = self._norm_date(date_txt)
            list_date_obj = self._to_date(list_date_str)

            # 若列表日期存在且「早於 cutoff」，因列表本身是新→舊排序，可直接停止處理本頁剩餘卡片
            if list_date_obj and list_date_obj < self.cutoff:
                stop_pagination = True
                # 不處理更舊的卡片
                continue

            url = self._force_https(response.urljoin(href))
            path = urllib.parse.urlparse(url).path or ""
            if self.DETAIL_RE.search(path):
                any_kept = True
                yield response.follow(
                    self._ensure_lang_zh(url),
                    callback=self.parse_detail,
                    meta={"list_date": list_date_str},
                )

        # 2) 是否需要抓下一頁？
        #    - 若本頁就已遇到「舊於 cutoff」→ 不再翻頁
        #    - 若本頁都在範圍內，且仍有下一頁 → 繼續
        # 取得目前頁碼與總頁數
        m_total = re.search(r"totalPage\s*:\s*(\d+)", response.text)
        total_pages = int(m_total.group(1)) if m_total else 1
        m_curr = self.PAGE_RE.search(response.url)
        curr_page = int(m_curr.group(1)) if m_curr else 1

        if not stop_pagination and any_kept and (curr_page < total_pages):
            next_url = f"https://cse.ttu.edu.tw/p/403-1058-61-{curr_page + 1}.php?Lang=zh-tw"
            yield response.follow(next_url, callback=self.parse)

    def parse_detail(self, response):
        sel: Selector = response.selector

        title = (sel.css("meta[property='og:title']::attr(content)").get()
                 or sel.css("title::text").get() or "").strip() or None
        h1 = (sel.css("h1::text").get() or "").strip() or None

        # 內頁日期（優先），抓不到用列表日期
        published = (sel.css("time::attr(datetime)").get()
                     or sel.css("time::text").get()
                     or sel.css(".mdate::text, .date::text, .post-date::text").get())
        published = self._norm_date(published) or response.meta.get("list_date")

        text = self._extract_main_text(response) or ""

        yield {
            "url": self._ensure_lang_zh(self._force_https(response.url)),
            "title": title,
            "h1": h1,
            "published_at": published,
            "content": text.strip(),
        }

    # -------- content extraction --------
    def _extract_main_text(self, response: scrapy.http.Response) -> str:
        """優先鎖定 .meditor；支援純文字 + <br> 版型，並排除頁首/頁尾雜訊"""
        # 1) 先把 <br> 轉換為換行，避免整段黏在一起
        html = re.sub(r"(?i)<br\s*/?>", "\n", response.text)
        sel = Selector(text=html)

        # 2) 內容容器：這站的內文固定在 .mpgdetail .meditor（或 .module-detail .meditor）
        content_area = sel.css(".mpgdetail .meditor, .module.module-detail .meditor")
        if not content_area:
            content_area = sel.css("main, article, #page_content, .content, .mcont") or sel

        # 3) 先試 Readability，但要求最小長度，避免只抓到標題
        try:
            from readability import Document
            doc = Document(html)
            s = Selector(text=doc.summary(html_partial=True))
            parts = s.xpath(".//p//text() | .//li//text() | .//h2//text() | .//h3//text()").getall()
            text_rd = "\n".join(t.strip() for t in parts if t and t.strip())
            text_rd = re.sub(r"\n{3,}", "\n\n", unescape(text_rd)).strip()
            # 不要回傳太短或疑似只有標題的結果
            if len(text_rd) > 40 and "大同大學 資訊工程學系" not in text_rd:
                return text_rd
        except Exception:
            pass

        # 4) 黑名單：排除頁首/頁尾/燈箱等
        noise_classes = [
            "hdmenu","mnavbar","navbar","mycollapse","iosScrollToggle",
            "hd-topnav2","topnav2","hd_search","breadcrumb","marquee",
            "pswp","toTop","footer","copyright","bt_main","bt_text","bt_share"
        ]
        noise_ids = ["Dyn_footer"]

        # 5) 核心：直接在 .meditor 範圍抓「所有文字」，包含直屬 text node
        #    （這就是解掉你三個案例的關鍵）
        texts = content_area.xpath(".//text()").getall()

        # 6) 清理與整形
        text = "\n".join(t.replace("\xa0", " ").strip() for t in texts if t and t.strip())
        # 去掉頁尾系辦資訊尾巴
        if text:
            lines = [ln for ln in text.split("\n") if ln.strip()]
            tail_keys = ["大同大學 資訊工程學系", "24小時緊急聯絡電話", "ext.6565"]
            while lines and any(k in lines[-1] for k in tail_keys):
                lines.pop()
            text = "\n".join(lines)

        # 壓縮多重空白與過多的空行
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text



